import torch


class JetEventGenerator:
    """
    Single-jet event generator.

    Approximation: daughters of a jet share a vertex upstream of the calorimeter
    with Gaussian-distributed angular offsets from the jet axis. Their projected
    impact points on the detector face are sampled directly from a 2D Gaussian
    centered at the jet axis. Each daughter is then passed to the simulator as
    if it were a vertical particle at its impact point — i.e., we ignore the
    small obliquity of each daughter's shower axis (an O(sigma_theta) effect,
    well below the cell pitch for typical jet cones and detector geometries).

    Per event we sample:
      - one jet axis (x_jet, y_jet) on the detector face, uniform in (x|y)_range
      - one total jet energy E_jet, power-law in E_range (matches EventGenerator)
      - N daughters, uniform in N_range
      - daughter energy fractions z_i from Dirichlet(alpha) (sum to 1 over actives)
      - daughter positions (x_i, y_i) ~ Normal((x_jet, y_jet), sigma_face_cm^2)

    Config fields (additional to the standard EventGenerator fields):
      jet_sigma_face_cm    : sigma of the daughter Gaussian on the detector face,
                             cm (default 0.7).
      frag_dirichlet_alpha : Dirichlet concentration for daughter energy
                             fractions (default 2.0). alpha > 1 disfavors a
                             single daughter carrying ~all of E_jet.
    Same E_range / N_range / power semantics as EventGenerator, but interpreted
    as TOTAL jet energy and daughters per jet.
    """

    def __init__(self, config, device=None):
        self.config = config
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']
        self.power = config.get('power', 2.0)
        self.pad_value = float(config.get('pad_value', 'nan'))
        self.sigma_face_cm = config.get('jet_sigma_face_cm', 0.7)
        # Dirichlet concentration. alpha=5 gives marginal Beta(5, 5*(N-1)),
        # which has very small probability of any single daughter having
        # fraction < 0.005 — this is needed to keep daughter energies above
        # the GFlash-stability danger zone (~500 MeV) where sigma_ln_alpha
        # becomes large enough that 3σ-low draws give degenerate gamma profiles.
        self.frag_alpha = float(config.get('frag_dirichlet_alpha', 5.0))
        self.set_device(device)

    def set_device(self, device):
        if device is None:
            self.device = self.config.get('device', 'cpu')
        else:
            self.device = device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def generate(self, N_events=1):
        # 1) jet axis impact point on the detector face, one per event
        x_jet = torch.rand(N_events, device=self.device) * (self.x_max - self.x_min) + self.x_min
        y_jet = torch.rand(N_events, device=self.device) * (self.y_max - self.y_min) + self.y_min

        # 2) total jet energy per event (power-law, same shape as EventGenerator)
        r = torch.rand(N_events, device=self.device)
        E_jet = ((self.E_max ** (1 - self.power) - self.E_min ** (1 - self.power)) * r
                 + self.E_min ** (1 - self.power)) ** (1 / (1 - self.power))

        # 3) daughters per event
        N_part = torch.randint(self.N_min, self.N_max + 1, (N_events,), device=self.device)
        N_pad = int(N_part.max().item())

        # 4) energy fractions from Dirichlet(alpha) over ACTIVE slots only.
        # Sample independent Gamma(alpha, 1), zero out inactive slots, then normalize.
        # The active fractions then sum to 1 per event (Dirichlet by construction).
        gamma = torch.distributions.Gamma(self.frag_alpha, 1.0).sample(
            (N_events, N_pad)).to(self.device)
        idx = torch.arange(N_pad, device=self.device).unsqueeze(0)   # (1, N_pad)
        active = idx < N_part.unsqueeze(1)                           # (B, N_pad)
        gamma_active = gamma * active.float()
        z = gamma_active / gamma_active.sum(dim=1, keepdim=True).clamp(min=1e-12)
        particle_Es = z * E_jet.unsqueeze(1)                         # (B, N_pad)

        # 5) daughter positions on the detector face — Gaussian around jet axis
        dx = torch.randn(N_events, N_pad, device=self.device) * self.sigma_face_cm
        dy = torch.randn(N_events, N_pad, device=self.device) * self.sigma_face_cm
        particle_xs = x_jet.unsqueeze(1) + dx
        particle_ys = y_jet.unsqueeze(1) + dy

        # 6) padded slots get pad_value (NaN or 0 depending on config)
        mask = ~active
        particle_Es[mask] = self.pad_value
        particle_xs[mask] = self.pad_value
        particle_ys[mask] = self.pad_value

        return particle_Es, particle_xs, particle_ys


class EventGenerator:

    def __init__(self, config, device=None):
        self.config = config
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']
        self.power = config.get('power', 2.0)
        self.pad_value = float(config.get('pad_value', 'nan'))
        self.set_device(device)

    def set_device(self, device):
        if device is None:
            self.device = self.config.get('device', 'cpu')
        else:
            self.device = device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def generate(self, N_events=1):

        N_particles = torch.randint(self.N_min, self.N_max + 1, (N_events,), device=self.device)

        N_pad = N_particles.max().item()

        ### Uniform spread in x/y
        particle_xs = torch.rand((N_events, N_pad), device=self.device) * (self.x_max - self.x_min) + self.x_min
        particle_ys = torch.rand((N_events, N_pad), device=self.device) * (self.y_max - self.y_min) + self.y_min

        ### Power-law spectrum in energy
        r = torch.rand((N_events, N_pad), device=self.device)
        particle_Es = ((self.E_max**(1-self.power) - self.E_min**(1-self.power)) * r + self.E_min**(1-self.power))**(1/(1-self.power))

        if N_events > 1:
            ### Assign pad_value to padded particles
            index = torch.arange(N_pad, device=self.device).unsqueeze(0)  # (1, N_pad)
            mask = index >= N_particles.unsqueeze(1)  # (N_events, N_pad)
            particle_Es[mask] = self.pad_value
            particle_xs[mask] = self.pad_value
            particle_ys[mask] = self.pad_value

        return particle_Es, particle_xs, particle_ys