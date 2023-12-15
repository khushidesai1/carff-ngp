import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image
import torch.nn.functional as F
from nerf.utils import *
import torchmetrics
from matplotlib import pyplot as plt 
import traceback

hidden_dim = 512
K = 2  # Number of mixtures

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

        # wrong: rotate along global x/y axis
        #self.rot = R.from_euler('xy', [-dy * 0.1, -dx * 0.1], degrees=True) * self.rot
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

        # wrong: pan in global coordinate system
        #self.center += 0.001 * np.array([-dx, -dy, dz])
    


class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, input_path="train/cam-v0-t2", debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        # self.latent_toggle = True
        self.training = True
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        # index = self.find_index("train/cam-v0-t2")
        index = self.find_index(input_path)
        mu = train_loader._data.mus[index]
        var = train_loader._data.vars[index]
        self.test_latent = reparameterize(mu, var)
        self.current_t = 0
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        dpg.create_context()
        # self.register_dpg()
        self.test_step()

        latent_dim = self.test_latent.shape[0]
        self.MDN = MDN(2*latent_dim,latent_dim, K, hidden_dim).cuda()
        self.MDN.load(self.train_loader._data.root_path)
        print("MDN Path:", self.train_loader._data.root_path)


    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        # dpg.set_value("_log_train_time", f'{t:.4f}ms')
        # dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    
    def test_step(self):
        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            latent = self.test_latent
            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, latent, self.bg_color, self.spp, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = outputs['image']
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + outputs['image']) / (self.spp + 1)
                self.spp += 1

    def render_bev(self, dest_path='rendered.png'):
        '''
        for terminal viz, renders the bev image
        '''
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        # print(indices)
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']

        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1)
        pred_img = torch.from_numpy(outputs['image']) #.reshape(-1, H, W, 3)
        save_image(pred_img.permute(2, 0, 1), dest_path)
        plt.imshow(pred_img)
        return outputs
    
    def calculate_densities(self, target_positions):
        '''
        for terminal viz, renders the bev image with target positions marked in blue
        '''
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        # print(indices)
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']

        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1, target_positions=target_positions)
        return outputs['mean_density']

    def update_and_compute_densities(self, image_path, target_positions1, target_positions2, dest_path='rendered.png'):
        result_index = self.find_index(image_path)
        # update test_latent
        if result_index == -1:
            print("Error: input image path invalid for latent generation")
            return
        # update test_latent
        print("Index:", result_index)
        mus = self.train_loader._data.mus[result_index].cuda()
        vars = self.train_loader._data.vars[result_index].cuda()
        # one hot encode
        input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
        sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
        predicted_latent = sampled_latent.squeeze(0)
        self.test_latent = predicted_latent
        self.need_update = True
        print("Success: changed to PREDICTED latents generated by ", image_path)
        return self.calculate_densities(target_positions1), self.calculate_densities(target_positions2)

    def find_index(self, input):
        '''
        helper method to find the corresponding index of the input path
        '''
        indices = [index for index, path in enumerate(self.train_loader._data.paths) if input in path]
        if len(indices) != 1:
            return -1
        print("Input corresponds to:", self.train_loader._data.paths[indices[0]])
        return indices[0]

    def probe(self):
        '''
        helper method, simple img based probe
        '''
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        # print(indices)
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']

        # see what it looks like with current latent
        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1)
        pred_img = torch.from_numpy(outputs['image']) #.reshape(-1, H, W, 3)
        save_image(pred_img.permute(2, 0, 1), 'current_probe_img.png')

        # get all images at the same view and compare
        losses = {}
        for key in indices.keys():
            index = indices[key]
            this_gt = self.train_loader._data.collate_for_probe([index])["images"].squeeze(0)
            losses[key] = F.mse_loss(this_gt.cuda(), pred_img.cuda())
        probed_result = min(losses, key=losses.get)
        # get the distribution for that result
        return indices[probed_result]

    def update_latent_from_image(self, image_path, dest_path='rendered.png'):
        index = self.find_index(image_path)
        # update test_latent
        if index == -1:
            print("Error: input image path invalid for latent generation")
            return
        mu = self.train_loader._data.mus[index]
        var = self.train_loader._data.vars[index]
        new_latent = reparameterize(mu, var)
        self.test_latent = new_latent
        self.need_update = True
        print("Success: changed to latents generated by ", image_path)
        return self.render_bev(dest_path=dest_path)

    def update_latent_from_predicted(self, image_path, dest_path='rendered.png'):
        result_index = self.find_index(image_path)
        # update test_latent
        if result_index == -1:
            print("Error: input image path invalid for latent generation")
            return
        # update test_latent
        print("Index:", result_index)
        mus = self.train_loader._data.mus[result_index].cuda()
        vars = self.train_loader._data.vars[result_index].cuda()
        # one hot encode
        input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
        sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
        predicted_latent = sampled_latent.squeeze(0)
        self.test_latent = predicted_latent
        self.need_update = True
        print("Success: changed to PREDICTED latents generated by ", image_path)
        return self.render_bev(dest_path=dest_path)


    def render(self, epoch):
        for _ in tqdm(range(epoch)):
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
