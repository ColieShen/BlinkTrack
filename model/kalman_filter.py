import torch
from torch import nn


class kalman_filter_4(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=128):
        super(kalman_filter_4, self).__init__()

        self.prev_ts = 0

    def reset(self):
        self.prev_ts = 0

    def predict(self, prev_state, ts):
        prev_x = prev_state['filter_state']['x']
        prev_pred_state = prev_state['predict']
        P = prev_state['filter_state']['P']
        Q = prev_state['filter_state']['Q']
        B, N = prev_x.shape[0], prev_x.shape[1]

        # State transition matrix
        dt = ts - self.prev_ts
        self.prev_ts = ts
        Fm = torch.tensor([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=P.device).expand(B, N, 4, 4)
        Fm.requires_grad = False

        pred_state = prev_state

        # state prediction
        x = Fm @ prev_x
        pred_state['filter_state']['x'] = x

        # Covariance prediction
        FPF = Fm @ P @ Fm.transpose(-1, -2)
        P = FPF + Q
        pred_state['filter_state']['P'] = P

        return x[..., :2, 0].clone(), pred_state, P[..., :2, :2]

    def update(self, flow, R, prev_state, ts):
        _, pred_state, _ = self.predict(prev_state, ts)

        pred_x = pred_state['filter_state']['x']    # B, N, 4, 1
        H = pred_state['filter_state']['H']  # fixed # B, N, 2, 4
        P = pred_state['filter_state']['P']  # B, N, 4, 4
        B, N = pred_x.shape[0], pred_x.shape[1]

        z = flow[..., None]  # B, N, 2, 1

        # Compute Kalman Gain
        # R     B, N, 2, 2
        S = H @ P @ H.transpose(-1, -2) + R  # B, N, 2, 2
        K = P @ H.transpose(-1, -2) @ torch.linalg.pinv(S)  # B, N. 4, 2

        # Update the estimate via measurement
        y = z - H @ pred_x  # Measurement residual  # B, N, 2, 1
        update_x = pred_x + K @ y

        # Update error covariance
        P = P - K @ H @ P

        update_state = pred_state
        update_state['filter_state']['x'] = update_x
        update_state['filter_state']['P'] = P

        return update_x[..., :2, 0].clone(), update_state, S[..., :2, :2], P[..., :2, :2]

    def detach(self, state):
        # state['filter_state']['F'] = state['filter_state']['F'].detach()
        state['filter_state']['H'] = state['filter_state']['H'].detach()
        state['filter_state']['x'] = state['filter_state']['x'].detach()
        state['filter_state']['P'] = state['filter_state']['P'].detach()
        state['filter_state']['Q'] = state['filter_state']['Q'].detach()

        return state


def init_filter_state(B, N, dt=1):
    # Measurement matrix
    H = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=torch.float32).expand(B, N, 2, 4)
    H.requires_grad = False

    # Initialize state and covariance matrix
    x = torch.zeros((B, N, 4, 1))
    x.requires_grad = False

    P = torch.eye(4, dtype=torch.float32).expand(B, N, 4, 4) * 10
    P.requires_grad = False

    dt = 1.0  # time step
    q = 1 ** 2  # assume q is the same in all directions
    Q = q * torch.tensor([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2],
    ], dtype=torch.float32).expand(B, N, 4, 4)
    Q.requires_grad = False

    return {'H': H, 'x': x, 'P': P, 'Q': Q}
