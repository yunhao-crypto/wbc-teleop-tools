"""Vita02A parallel-ankle kinematics for motor_theta policies.

Closed-loop policies observe and command the two Achilles motor angles in the
ankle pitch/roll slots. Body31 feedback and references remain virtual pitch/roll;
command slots follow the explicitly matched EXTERNAL ankle_command_space. ``motor_angles`` and ``jacobian`` port
``gear_sonic.utils.parallel_ankle`` (the training map, also used by
vita-robot's ``GrootMotorThetaAnkleMap``); ``forward`` inverts the map with
Newton iterations like the vita-robot ankle adapter.

Conventions: ``roll`` below is the mechanism roll, ``roll_sign * virtual
roll`` (left +1, right -1). Jacobian columns are ``(roll, pitch)``, rows
``(theta_top, theta_bottom)``.
"""
import math

import numpy as np

SIDES = ('left', 'right')


def _foot(pitch, roll, x, y, z):
    """R_y(pitch) @ R_x(roll) @ (x, y, z)."""
    cr, sr = math.cos(roll), math.sin(roll)
    y1, z1 = cr*y-sr*z, sr*y+cr*z
    cp, sp = math.cos(pitch), math.sin(pitch)
    return cp*x+sp*z1, y1, -sp*x+cp*z1


def _links(g):
    return ((g['ly'], g['z_top']), (-g['ly'], g['z_bot']))


def motor_angles(pitch, roll, g):
    """Mechanism (pitch, roll) to (theta_top, theta_bottom)."""
    lx, out = g['lx'], []
    for y, height in _links(g):
        ax, ay, az = _foot(pitch, roll, -lx, y, 0.)
        a, dy, b = ax, ay-y, az-height
        c = -(height*height-lx*lx-(a*a+dy*dy+b*b))/(2.*lx)
        disc = b*b*c*c-(a*a+b*b)*(c*c-a*a)
        s = (b*c+math.sqrt(max(disc, 0.)))/(a*a+b*b)
        out.append(math.asin(min(1., max(-1., s))))
    return out


def jacobian(pitch, roll, g, theta=None):
    """d(theta_top, theta_bottom)/d(roll, pitch) at a mechanism pose."""
    lx = g['lx']
    theta = motor_angles(pitch, roll, g) if theta is None else theta
    cp, sp = math.cos(pitch), math.sin(pitch)
    rows = []
    for (y, height), t in zip(_links(g), theta):
        arm = (-lx*math.cos(t), 0., lx*math.sin(t))
        crank = (arm[0], y, height+arm[2])
        anchor = _foot(pitch, roll, -lx, y, 0.)
        rod = tuple(p-q for p, q in zip(anchor, crank))
        den = arm[2]*rod[0]-arm[0]*rod[2]
        m = (anchor[1]*rod[2]-anchor[2]*rod[1], anchor[2]*rod[0]-anchor[0]*rod[2],
             anchor[0]*rod[1]-anchor[1]*rod[0])
        rows.append(((m[0]*cp-m[2]*sp)/den, m[1]/den))
    return rows


def forward(theta_top, theta_bottom, g, *, max_iter=50, tol=1e-10):
    """Mechanism (pitch, roll) whose motor angles are closest to the request.

    Damped Gauss-Newton from the neutral pose. Returns the pose and the
    remaining motor-angle residual; an unreachable request (outside the
    linkage workspace) ends at the nearest reachable pose.
    """
    pitch = roll = 0.
    target = (theta_top, theta_bottom)
    best = (math.inf, 0., 0.)
    for _ in range(max_iter):
        theta = motor_angles(pitch, roll, g)
        f0, f1 = theta[0]-target[0], theta[1]-target[1]
        residual = max(abs(f0), abs(f1))
        if residual < best[0]:
            best = (residual, pitch, roll)
        if residual < tol:
            break
        (a, b), (c, d) = jacobian(pitch, roll, g, theta)
        # (J^T J + mu I) delta = J^T f; mu only matters near singular poses.
        mu = 1e-12
        h00, h01, h11 = a*a+c*c+mu, a*b+c*d, b*b+d*d+mu
        g0, g1 = a*f0+c*f1, b*f0+d*f1
        det = h00*h11-h01*h01
        d_roll, d_pitch = (h11*g0-h01*g1)/det, (h00*g1-h01*g0)/det
        step = max(abs(d_roll), abs(d_pitch))
        if step > .2:
            d_roll, d_pitch = d_roll*.2/step, d_pitch*.2/step
        roll, pitch = roll-d_roll, pitch-d_pitch
    return best[1], best[2], best[0]


class MotorThetaAnkle:
    """Serial/virtual <-> policy (motor_theta) ankle slots of a 28-joint vector."""

    def __init__(self, names, block):
        g = block.get('geometry', {})
        if (block.get('schema_version') != 3 or block.get('control_space') != 'motor_theta'
                or block.get('reference_conversion') != 'virtual_pitch_roll_to_motor_theta'):
            raise ValueError('unsupported motor_theta ankle block')
        self.geometry = {k: float(g[k]) for k in ('lx', 'ly', 'z_top', 'z_bot')}
        if not all(math.isfinite(v) and v > 0 for v in self.geometry.values()):
            raise ValueError('motor_theta ankle geometry must be finite and positive')
        if block.get('roll_sign') != {'left': 1, 'right': -1}:
            raise ValueError('motor_theta ankle roll_sign must be left=+1, right=-1')
        limits = [float(x) for x in block.get('motor_limits', ())]
        if len(limits) != 2 or not all(map(math.isfinite, limits)) or limits[0] >= limits[1]:
            raise ValueError('motor_theta ankle motor_limits must be a finite [lower, upper]')
        self.motor_limits = tuple(limits)
        expected = {f'{side}_ankle_{axis}_joint': dict(side=side, motor=motor)
                    for side in SIDES for axis, motor in (('pitch', 'theta_top'), ('roll', 'theta_bottom'))}
        slots = block.get('slots', {})
        if ({k: dict(side=v.get('side'), motor=v.get('motor')) for k, v in slots.items()} != expected):
            raise ValueError('motor_theta ankle slots must map pitch->theta_top and roll->theta_bottom')
        names = list(names)
        self.legs = tuple((names.index(f'{side}_ankle_pitch_joint'), names.index(f'{side}_ankle_roll_joint'),
                           block['roll_sign'][side]) for side in SIDES)
        self.slots = np.array([i for leg in self.legs for i in leg[:2]])

    def to_policy(self, q):
        """Virtual pitch/roll ankle slots of ``q`` ([..., 28]) to motor angles."""
        q = np.array(q, dtype=np.float64)
        flat = q.reshape(-1, q.shape[-1])
        for row in flat:
            for p, r, sign in self.legs:
                row[p], row[r] = motor_angles(row[p], sign*row[r], self.geometry)
        return q

    def velocity_to_policy(self, q, qd):
        """Virtual ankle rates to motor rates through the Jacobian at ``q``."""
        q, qd = np.asarray(q, dtype=np.float64), np.array(qd, dtype=np.float64)
        for pose, rate in zip(q.reshape(-1, q.shape[-1]), qd.reshape(-1, qd.shape[-1])):
            for p, r, sign in self.legs:
                (a, b), (c, d) = jacobian(pose[p], sign*pose[r], self.geometry)
                droll, dpitch = sign*rate[r], rate[p]
                rate[p], rate[r] = a*droll+b*dpitch, c*droll+d*dpitch
        return qd

    def torque_to_virtual(self, q, tau):
        """Motor torques in the ankle slots of ``tau`` to virtual pitch/roll torques at pose ``q``.

        J^T at the virtual pose; vita-robot's ankle adapter applies the
        inverse (J^-T at the same pose) to a Body31 ankle torque.
        """
        q, tau = np.asarray(q, dtype=np.float64), np.array(tau, dtype=np.float64)
        for p, r, sign in self.legs:
            (a, b), (c, d) = jacobian(q[p], sign*q[r], self.geometry)
            top, bottom = tau[p], tau[r]
            tau[r], tau[p] = sign*(a*top+c*bottom), b*top+d*bottom
        return tau

    def to_virtual(self, q):
        """Motor-angle ankle slots of one 28-vector to virtual pitch/roll.

        Returns the vector and the largest motor-angle residual of the
        forward kinematics (zero up to tolerance inside the workspace).
        """
        q = np.array(q, dtype=np.float64)
        residual = 0.
        for p, r, sign in self.legs:
            pitch, roll, error = forward(q[p], q[r], self.geometry)
            q[p], q[r] = pitch, sign*roll
            residual = max(residual, error)
        return q, residual


def command_space(policy):
    """Wire coordinate contract, independent of the policy observation space."""
    value = policy.profile.get('ankle_command_space', 'virtual_pitch_roll')
    if value == 'policy':
        value = 'motor_theta' if getattr(policy.contract, 'ankle', None) is not None else 'virtual_pitch_roll'
    if value not in ('motor_theta', 'virtual_pitch_roll'):
        raise ValueError('ankle_command_space must be policy, motor_theta or virtual_pitch_roll')
    if value == 'motor_theta' and getattr(policy.contract, 'ankle', None) is None:
        raise ValueError('native motor commands require a motor_theta model contract')
    return value


def decode_motor_theta(policy, math_api, raw_action, **timing):
    """Body31 command of a motor_theta policy (SDK ``WbcTrackingPolicy`` subclass).

    Native mode sends A/B targets and motor gains directly. In compatibility
    mode, motor targets are mapped back to virtual pitch/roll through FK.
    The robot's motor PD (EXTERNAL ``ankle_virtual_pd_intent=false``) re-solves
    the same IK, which reproduces the trained motor PD inside the virtual
    ankle box. Outside it the robot clamps the virtual target first; with
    ``ankle_clamp_feedforward`` the motor PD torque that clamp removes is sent
    back as virtual ankle torque.
    """
    c = policy.contract
    action = np.asarray(raw_action, dtype=np.float64)
    if action.shape != (28,) or not np.isfinite(action).all():
        raise ValueError('model output must have 28 finite actions')
    policy.last_action = np.clip(action, -c.action_clip, c.action_clip).astype(np.float32)
    raw_q = c.policy_default+c.scale*policy.last_action.astype(np.float64)
    # Motor angles stop at the policy's motor travel, as in the vita-robot
    # passthrough. Only the compatibility path also clamps a virtual pose.
    motor_q = raw_q.copy()
    slots = c.ankle.slots
    motor_q[slots] = np.clip(raw_q[slots], c.policy_lower[slots], c.policy_upper[slots])
    native = command_space(policy) == 'motor_theta'
    tau = np.zeros(28)
    residual = None
    if native:
        # q/kp/kd stay in A/B space. Virtual limits and virtual clamp torque
        # compensation do not belong to this wire contract.
        target = np.clip(motor_q, policy.lower, policy.upper)
        target[slots] = motor_q[slots]
        clamped = target != raw_q
    else:
        virtual_q, residual = c.ankle.to_virtual(motor_q)
        target = np.clip(virtual_q, policy.lower, policy.upper)
        tau = np.zeros(28)
        if policy.profile['ankle_clamp_feedforward']:
            # Motor angles the robot will track: IK of the clamped virtual
            # target, then its motor-limit clamp (vita-robot PD-A).
            tracked = np.clip(c.ankle.to_policy(target)[slots], *c.ankle.motor_limits)
            missing = np.zeros(28)
            missing[slots] = c.kp[slots]*(motor_q[slots]-tracked)
            tau[slots] = c.ankle.torque_to_virtual(target, missing)[slots]
        clamped = target != virtual_q
        clamped[slots] |= motor_q[slots] != raw_q[slots]
    motors = [math_api.BodyJointCmd() for _ in range(31)]
    for p, slot in enumerate(policy.mapping):
        motors[slot] = math_api.BodyJointCmd(float(np.float32(target[p])), 0., float(np.float32(tau[p])),
                                             float(np.float32(c.kp[p])), float(np.float32(c.kd[p])))
    for name, fixed in policy.profile['fixed_joints'].items():
        motors[policy.order.index(name)] = math_api.BodyJointCmd(fixed['q'], 0., 0., fixed['kp'], fixed['kd'])
    return math_api.BodyCmd(joints=tuple(motors), **timing), {
        'ankle_command_space': command_space(policy),
        'action': policy.last_action.tolist(), 'raw_target': raw_q.tolist(),
        'motor_target': motor_q[slots].tolist(), 'ankle_fk_residual': residual,
        'target': target.tolist(), 'tau_ff': tau.tolist(), 'clamped': clamped.tolist(),
    }
