import torch
import torch.nn.functional as F

class SimpleSegGradCAM:
    """Grad-CAM for a segmentation model.
    We take the mean of the sigmoid logits as the target scalar.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.activ = None
        self.grads = None
        self.hook_a = self.layer.register_forward_hook(self._store_activ)
        self.hook_g = self.layer.register_full_backward_hook(self._store_grads)

    def _store_activ(self, module, inp, out):
        self.activ = out.detach()

    def _store_grads(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.sigmoid(logits)
        target = probs.mean()
        target.backward()

        weights = self.grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activ).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(2,3), keepdim=True)
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam, probs

    def close(self):
        self.hook_a.remove()
        self.hook_g.remove()
