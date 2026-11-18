"""Git ca0aa6e run_live body, unchanged; explicit modality reference hooks only.

No module monkeypatching. The original guard/startup/publish/exit code below is
copied verbatim and its source hash is tested against the fetched Git commit.
"""
from .actor_pico_live import *
from .pico_aligned_runtime import manual_reference,reference_yaw,pico_reference_command,GuardedInputs as Inputs

def run_live(runtime, args, write, controller, pico, stop, service_alive,
             clock=time, manual_pico=None, poll_command=lambda:None,
             request_action=None, motion_source=None):
    """One trial only. Every attempted entry is followed by checked CANCEL.

    Fake I/O in unit tests exercises this same runner; actual transport and
    hardware performance remain separate acceptance checks.
    """
    if not math.isfinite(args.seconds) or args.seconds < 0:
        raise ValueError('Live duration must be finite and nonnegative; 0 means until STOP')
    params, receipt = load_robot_config(args.robot_config, runtime)
    if motion_source is not None:
        write(dict(record_type='robot_configuration', robot_config=receipt, actions_sent=0))
    envelope = CommandEnvelope(runtime, controller.get_robot_info(args.startup_timeout), params)
    pico_max_age = runtime.config['pico_max_age_s']
    pico_loss_action = runtime.config.get('pico_loss_action', 'stand')
    auto = bool(getattr(args, 'auto_start', False))
    if getattr(args, 'entry_policy', 'policy-online') != 'policy-online':
        raise ValueError('Only policy-online startup is supported; joint-ramp was removed')
    wait_pico = bool(getattr(args, 'wait_pico', False))
    auto_pico = bool(getattr(args, 'auto_pico', False))
    startup = None
    if auto:
        from .actor_pico_startup import (StartupPlan, fixed_stand_startup,
                                         cancel_startup, FirmwareActions)
        startup = StartupPlan(runtime, envelope, args.robot_config)
    manual = bool(getattr(args, 'manual_pico', False))
    if motion_source is not None and (manual or wait_pico or auto_pico or args.stand_only):
        raise ValueError('File motion cannot combine Pico following or stand-only mode')
    requires_pico = motion_source is None and not args.stand_only and not manual
    if auto and ((not manual and motion_source is None) or args.stand_only):
        raise ValueError('One-click startup requires manual Pico mode')
    # Firmware retires completed keyframes, leaving current_action empty. Auto
    # explicitly requests FIXED_STAND and verifies measured targets afterward;
    # an empty name alone is never treated as completed fixed stand.
    initial_actions = ('PASSIVE', 'QUIET_PASSIVE', 'FIXED_STAND', '') if auto else ('PASSIVE',)
    inputs = Inputs(runtime, controller, pico if requires_pico else None, clock)
    guard = StatusGuard(clock, initial_actions=initial_actions)
    if manual and manual_pico is None:
        raise ValueError('Manual Pico mode requires an idle capture worker')
    gripper_config = getattr(args, 'gripper_config', None)
    if gripper_config is not None:
        if not manual or args.stand_only or motion_source is not None:
            raise ValueError('PICO gripper control requires live manual Pico mode without stand-only or file motion')
        if len(getattr(manual_pico, 'controls', None) or ()) < 7:
            raise ValueError('PICO gripper control requires the 7-slot controller array (grip snapshots)')
        from .actor_pico_gripper import GripperRunner, load_config as load_gripper_config
        gripper_config = load_gripper_config(gripper_config)
    runtime.reset_inputs()
    # Warm actual inference before taking ownership; erase its action history.
    runtime.infer_motion(runtime.hold_motion(),
        q=runtime.model.contract.default, dq=np.zeros(28), root=[1, 0, 0, 0], gyro=np.zeros(3))
    runtime.reset_inputs()
    preflight_deadline = clock.monotonic()+args.startup_timeout
    while not stop.is_set():
        if requires_pico and not service_alive():
            raise RuntimeError('Pico service exited during preflight')
        if clock.monotonic() >= preflight_deadline:
            raise TimeoutError('Preflight did not receive fresh status, Body31 and complete Pico window')
        try:
            state, q, dq, root, gyro = inputs.read(min(.1, max(.001, preflight_deadline-clock.monotonic())))
        except TimeoutError:
            clock.sleep(.02)
            continue
        status = controller.status
        if status is not None:
            guard.check(status)
        if (not requires_pico or inputs.reference_ready()) and status is not None:
            if requires_pico:
                inputs.require_pico()
            if auto:
                startup.check_hardware_pose(envelope, state)
            else:
                envelope.check_measured_pose(state)
            if auto:
                # The real first-command preflight belongs after FIXED_STAND.
                # A second policy rollout on the earlier PASSIVE pose is unused.
                break
            started = clock.monotonic()
            if requires_pico:
                command, obs, row = pico_reference_command(runtime, inputs.snapshot, started,
                    q=q, dq=dq, root=root, gyro=gyro)
            else:
                motion = runtime.hold_motion()
                motion = motion.lead_in(q, root, runtime.config['body_profile']['lead_in'])
                command, obs, row = runtime.infer_motion_command(motion,
                    q=q, dq=dq, root=root, gyro=gyro, yaw_offset=runtime.math.yaw(root))
            envelope.check(command)
            record_period_overrun(write,'preflight',clock.monotonic()-started,0)
            inputs.fresh(state)
            write(dict(row, record_type='live_preflight', robot_id=args.robot_id,
                       robot_config=receipt, status=status_row(status)), obs)
            break
        clock.sleep(runtime.model.contract.period)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Display and record actual targets/gains; starting live requests motion
    # directly, without a second typed MOVE prompt.
    print('ACTOR_LIVE_REVIEW '+json.dumps(dict(robot_id=args.robot_id,
        policy=runtime.policy_name, model=runtime.config['artifacts']['model'], config=receipt,
        stand_only=args.stand_only, seconds=args.seconds,
        manual_pico=manual, auto_start=auto, entry_policy='policy-online',
        reference_source='npz' if motion_source is not None else 'pico',
        motion=motion_source.metadata if motion_source is not None else None,
        wait_pico=wait_pico, auto_pico=auto_pico,
        startup_plan=startup.review if auto else None,
        default_pose=dict(zip(runtime.model.contract.names, runtime.model.contract.default.tolist())),
        kp=dict(zip(runtime.model.contract.names, runtime.model.contract.kp.tolist())),
        kd=dict(zip(runtime.model.contract.names, runtime.model.contract.kd.tolist())),
        fixed_joints=runtime.config['body_profile']['fixed_joints'],
        reference_delay_s=runtime.reference.delay if motion_source is None else 0.,
        human_height_m=runtime.config['retargeting']['human_height_m'] if motion_source is None else None,
        gmr_calibration=runtime.gmr_calibration if motion_source is None else None,
        pico_loss_action=pico_loss_action if motion_source is None else None,
        pico_stream=runtime.config.get('pico_stream') if motion_source is None else None,
        pico_worker=runtime.config.get('pico_worker'),
        head_control=runtime.config.get('head_control') if runtime.head is not None else None)), flush=True)
    print('ACTOR_START: motion requested; '+
          ('running until STOP/Ctrl-C or fault' if args.seconds == 0 else f'duration limit {args.seconds:g}s')+
          '; exit requests CANCEL -> PASSIVE. Keep support and E-stop available.',flush=True)
    if stop.is_set():
        return dict(mode='live', actions_sent=0, control_handles_created=1, entered=False)
    # Refresh effective configuration before requesting any motion.
    if load_robot_config(args.robot_config, runtime)[1] != receipt:
        raise RuntimeError('Effective config file changed during preflight')
    if auto:
        if not fixed_stand_startup(startup, envelope, runtime, args, controller, inputs,
                                  stop, write, clock, request_action=request_action,
                                  poll_command=poll_command):
            return dict(mode='live', actions_sent=0, entered=False, stopped_by_signal=True)
    startup_generation = controller.status.generation if auto else None
    cancel_action = (request_action or FirmwareActions(stop, clock, lambda:inputs.read(.1))) if auto else None
    try:
        if auto and load_robot_config(args.robot_config, runtime)[1] != receipt:
            raise RuntimeError('Effective config file changed during startup')
        runtime.reset_inputs()
        state, q, dq, root, gyro = inputs.read(STATE_AGE)
        if auto:
            startup.check_hardware_pose(envelope, state)
        else:
            envelope.check_measured_pose(state)
        if requires_pico:
            inputs.require_pico()
        guard = StatusGuard(clock, initial_actions=('FIXED_STAND', '') if auto else ('PASSIVE',))
        guard.check(controller.status)
        if auto:
            # PASSIVE can have a very different pose. Validate the first policy
            # command on the actual prepared FIXED_STAND feedback, not on an
            # earlier passive snapshot, and still before requesting EXTERNAL.
            runtime.reset_inputs()
            begin = clock.monotonic()
            lead = runtime.hold_motion().lead_in(
                q, root, runtime.config['body_profile']['lead_in'])
            command, obs, row = runtime.infer_motion_command(lead,q=q,dq=dq,root=root,gyro=gyro,
                                                            yaw_offset=runtime.math.yaw(root))
            model_elapsed = clock.monotonic()-begin
            # Preserve the actual failing input before any guard can raise.
            # This is one pre-ownership record, not a streaming log on the
            # 50 Hz control path. It supports exact inference reproduction.
            write(dict(row,record_type='prepared_policy_candidate',actions_sent=0,
                body_targets=[dict(q=j.q,dq=j.dq,tau=j.tau,kp=j.kp,kd=j.kd) for j in command.joints],
                input_state=dict(q=q.tolist(),dq=dq.tolist(),root=root.tolist(),gyro=gyro.tolist(),
                                 body_q=state.q.tolist(),state_id=int(state.state_id)),
                reference=dict(q=lead.q.tolist(),base=lead.base.tolist(),quat=lead.quat.tolist(),
                               ref_time=0.,yaw_offset=runtime.math.yaw(root)),
                observation=obs.tolist()))
            envelope.check(command)
            record_period_overrun(write,'prepared_preflight',model_elapsed,0)
            inputs.fresh(state)
            write(dict(row,record_type='prepared_policy_preflight',actions_sent=0),obs)
            runtime.reset_inputs()
    except BaseException:
        if auto:
            cancel_startup(controller, cancel_action, clock, write, startup_generation)
        raise
    attempted, sent_count = False, 0
    report = dict(mode='live', actions_sent=0, control_handles_created=1, entered=False,
                  exit_confirmed=False, robot_id=args.robot_id, stand_only=args.stand_only,
                  auto_start=auto, onnx_control_enabled=False,
                  missed_policy_ticks=0, duration_limit_s=args.seconds, pico_losses=0)
    grippers = None

    def write_gripper_events():
        for row in grippers.drain():
            write(dict(row, actions_sent=sent_count))

    try:
        if stop.is_set():
            return report
        if gripper_config is not None:
            # Info, State and output preparation complete before entry; output
            # permission is only checked once the gripper thread starts.
            grippers = GripperRunner(gripper_config, manual_pico.controls, connection=controller)
            write_gripper_events()
        attempted = True
        entered = controller.enter_external(timeout=5.)
        if (entered.current_action != 'EXTERNAL' or phase(entered) != 'RUNNING'
                or entered.client_id != controller.client_id or not entered.control_id
                or entered.generation <= 0):
            raise RuntimeError('Entry did not confirm owned EXTERNAL/RUNNING')
        report['entered'] = True
        write(dict(record_type='external_entered', status=status_row(entered), actions_sent=0))
        if grippers is not None:
            grippers.start()
            print('PICO_GRIPPER: hold a side grip to close that gripper, release to open; '
                  'lost or stale input opens.', flush=True)
        runtime.reset_inputs()
        state, q, dq, root, gyro = inputs.read(0)
        if requires_pico:
            inputs.require_pico()
        if auto:
            startup.check_hardware_pose(envelope,state)
        if not auto:
            envelope.check_measured_pose(state)
        lead = runtime.hold_motion().lead_in(
            q, root, runtime.config['body_profile']['lead_in'])
        offset = runtime.math.yaw(root)
        preparation = max(3., (len(lead.q)-1)*runtime.model.contract.period)
        start = deadline = clock.monotonic()
        end = start+args.seconds if args.seconds > 0 else math.inf
        policy_tick = 0
        stand_tick = 0
        tracking = False
        pico_losses = 0
        pico_reengage_blocked = False
        motion_complete = False
        if runtime.head is not None and not args.stand_only and motion_source is None:
            print('PICO_HEAD: look straight ahead relative to your chest when enabling following; '
                  'this pose becomes head neutral.', flush=True)
        last_inferred_identity = None
        last_inferred_receipt_ns = None
        report['duplicate_body_ticks'] = 0
        if auto:
            print('POLICY_LEAD_IN: ONNX runs from measured-pose reference toward model stand; '+
                  ('then fixed NPZ playback' if motion_source is not None else 'awaiting Pico'),flush=True)
        if manual:
            print('PICO_BUTTONS: release both triggers, then squeeze both past 0.7 to toggle '
                  'following/ONNX stand. STOP or Ctrl-C cancels.',flush=True)
        previous_pico_state = None
        previous_controller_state = None
        last_trigger_event = 0
        modes=getattr(runtime,'teleop_modes',None)
        if modes is not None:
            print('TELEOP_MODE: full_body; center and click right stick to switch. Right Y=forward/back, X=turn in joystick mode.',flush=True)
        if manual and wait_pico:
            manual_pico.connect(root)
            print('PICO_WAIT: receiver starting; ONNX stand does not depend on receiving Pico',flush=True)
            write(dict(record_type='pico_connect_requested',automatic=True,actions_sent=0))
        stale_ticks = 0
        logged_drops = entered.ingress_drop_count

        def skip_stale(phase_name, error, **details):
            nonlocal stale_ticks
            stale_ticks += 1
            cancel = stale_ticks > MAX_STALE_TICKS
            write(dict(record_type='body_state_age_failure', phase=phase_name,
                       consecutive_ticks=stale_ticks, action='cancel' if cancel else 'skip_tick',
                       actions_sent=sent_count, error=str(error), **details))
            if cancel:
                raise StaleStateError(f'BodyState stale for {stale_ticks} consecutive policy ticks: {error}')

        def handle_pico_loss(error, q, root):
            """Pico lost while following: return to ONNX stand, same EXTERNAL session.

            Only an unusable Pico reference during tracking takes this path;
            robot-feedback, firmware and envelope faults still cancel. After a
            fault there is no automatic re-engagement: manual mode needs a fresh
            double-trigger (same as PICO_STANDBY), requires_pico mode keeps
            standing until STOP. q/root are the latest validated measurement.
            """
            nonlocal tracking, auto_pico, lead, offset, stand_tick, preparation
            nonlocal pico_losses, pico_reengage_blocked
            if not tracking or motion_source is not None or pico_loss_action != 'stand':
                return False
            tracking = False
            auto_pico = False
            if modes is not None:
                modes.cancel();modes.upper_follow=None
            pico_reengage_blocked = True
            # Required-Pico mode now stands using robot feedback only. A
            # terminal worker snapshot must not keep failing Inputs.read().
            # Manual re-engagement reads its separate manual_pico source.
            inputs.pico = None
            lead = runtime.hold_motion().lead_in(q, root, runtime.config['body_profile']['lead_in'])
            offset = runtime.math.yaw(root)
            stand_tick = 0
            preparation = max(.3, (len(lead.q)-1)*runtime.model.contract.period)
            pico_losses += 1
            report['pico_losses'] = pico_losses
            print('ACTOR_WARNING: Pico lost during following: '+str(error)+
                  '; returning to ONNX stand, EXTERNAL session stays active. '+
                  ('Re-engage with a fresh double-trigger once Pico recovers; '
                   if manual else 'No re-engagement this run; ')+
                  'STOP/Ctrl-C still cancels to PASSIVE.', file=sys.stderr, flush=True)
            write(dict(record_type='pico_loss_stand', error=str(error), loss_count=pico_losses,
                       transition_start_q=q.tolist(), policy_history_reset=False,
                       actions_sent=sent_count))
            return True

        def advance(current):
            following, missed = next_policy_deadline(current, clock.monotonic(), runtime.model.contract.period)
            report['missed_policy_ticks'] += missed
            if missed:
                write(dict(record_type='policy_ticks_skipped', missed_ticks=missed,
                           total_missed_ticks=report['missed_policy_ticks'], actions_sent=sent_count))
            return following

        while not stop.is_set() and clock.monotonic() < end:
            if motion_source is not None:
                poll_command()  # Terminal STOP/EOF sets the same cancellation event.
                if stop.is_set():
                    break
            if requires_pico and not service_alive() and not pico_reengage_blocked:
                error = RuntimeError('Pico service exited')
                if not handle_pico_loss(error, q, root):
                    raise error
                deadline = advance(deadline)
                continue
            clock.sleep(max(0., deadline-clock.monotonic()))
            if stop.is_set() or clock.monotonic() >= end:
                break
            cycle = clock.monotonic()
            cpu_clock = getattr(clock,'thread_time',None)
            cpu_start = cpu_clock() if cpu_clock is not None else None
            status = guard.check(controller.status, entered)
            if grippers is not None:
                write_gripper_events()
                grippers.check()
            if status.ingress_drop_count != logged_drops:
                write(dict(record_type='firmware_ingress_drop', previous=logged_drops,
                           count=status.ingress_drop_count, actions_sent=sent_count))
                logged_drops = status.ingress_drop_count
            status_end = clock.monotonic()
            try:
                state, q, dq, root, gyro = inputs.read(0)
            except StaleStateError as error:
                skip_stale('at_read', error)
                deadline = advance(deadline)
                continue
            except PicoWorkerError as error:
                # q/root are the previous tick's validated measurement here.
                if not handle_pico_loss(error, q, root):
                    raise
                deadline = advance(deadline)
                continue
            state_age_at_read = inputs.ages(state)
            state_end = clock.monotonic()
            identity = (int(state.state_id), int(state.ts_state_real_us))
            if identity == last_inferred_identity:
                # PullStateSource returns the cached snapshot when its keep-last
                # ring is empty. Do not advance model history/reference or issue
                # a new action from that same measurement. A republished heartbeat
                # may refresh receipt/publication clocks without new feedback;
                # bound it by the original receipt and the existing age budget.
                unchanged_ms = (round(clock.monotonic()*1e9)-last_inferred_receipt_ns)/1e6
                if unchanged_ms > STATE_AGE*1000:
                    skip_stale('at_read', StaleStateError(
                        f'BodyState identity did not advance: state_id={state.state_id}, '
                        f'unchanged_age_ms={unchanged_ms:.3f}, limit_ms={STATE_AGE*1000:g}'))
                else:
                    report['duplicate_body_ticks'] += 1
                    write(dict(record_type='body_state_not_advanced', state_id=int(state.state_id),
                        source_state_us=int(state.ts_state_real_us), ages=state_age_at_read,
                        unchanged_age_ms=unchanged_ms, action='skip_tick', actions_sent=sent_count))
                deadline = advance(deadline)
                continue
            if requires_pico and not pico_reengage_blocked:
                try:
                    inputs.require_pico()
                except TimeoutError as error:
                    if not handle_pico_loss(error, q, root):
                        raise
                    deadline = advance(deadline)
                    continue
            # Match SDK ReferenceSession: one reference frame per successful
            # inference/publication, independent of wall-clock scheduling jitter.
            elapsed = stand_tick*runtime.model.contract.period
            reference_frame = min(stand_tick,len(lead.q)-1)
            reference_time = reference_frame*runtime.model.contract.period
            if not tracking and getattr(runtime, 'planner', None) is not None and runtime.planner.active:
                runtime.planner.reset()
            snapshot = None
            if manual:
                request = poll_command()
                if request == 'connect':
                    if manual_pico.connect(root):
                        print('PICO_CONNECTING: ONNX stand continues; use the configured GMR reference stance.', flush=True)
                        write(dict(record_type='pico_connect_requested', actions_sent=sent_count))
                snapshot = manual_pico.snapshot()
                controller_age = clock.monotonic()-float(snapshot.get('controller_received_monotonic',-math.inf))
                controller_ready = 0 <= controller_age <= pico_max_age
                if controller_ready != previous_controller_state:
                    previous_controller_state = controller_ready
                    print('PICO_BUTTONS: '+('controller input ready; release then squeeze both triggers to toggle'
                        if controller_ready else 'waiting for fresh controller input'),flush=True)
                    write(dict(record_type='pico_controller_input',ready=controller_ready,actions_sent=sent_count))
                event = int(snapshot.get('trigger_event_seq',last_trigger_event))
                if event != last_trigger_event:
                    previous_event = last_trigger_event
                    last_trigger_event = event
                    event_age = clock.monotonic()-float(snapshot.get('trigger_event_monotonic',-math.inf))
                    controller_age = clock.monotonic()-float(snapshot.get('controller_received_monotonic',-math.inf))
                    if event == previous_event+1 and 0 <= event_age <= pico_max_age and 0 <= controller_age <= pico_max_age:
                        request = 'toggle'
                        write(dict(record_type='pico_trigger_toggle',event_seq=event,actions_sent=sent_count))
                    else:
                        write(dict(record_type='pico_trigger_rejected',reason='stale/missed trigger edge',
                                   event_seq=event,actions_sent=sent_count))
                state_name = snapshot.get('state')
                state_key = state_name
                if state_key != previous_pico_state:
                    previous_pico_state = state_key
                    detail = (': '+str(snapshot['error'])) if snapshot.get('error') else ''
                    print(f'PICO_INPUT: {state_name}{detail}; ONNX '+('tracking' if tracking else 'stand'), flush=True)
                    write(dict(record_type='pico_input_state', state=state_name,
                               error=snapshot.get('error'),
                               source_gap_s=snapshot.get('source_gap_s'),
                               receipt_gap_s=snapshot.get('receipt_gap_s'),
                               retarget_ms=snapshot.get('retarget_ms'),actions_sent=sent_count))
                if modes is not None:
                    # Pause has priority over a concurrent mode click. Consume
                    # rejected edges so held/cached buttons cannot re-engage.
                    if request=='toggle':
                        modes.cancel();modes.upper_follow=None
                    event=modes.request(snapshot,clock.monotonic(),tracking=tracking and request!='toggle',
                        q=q,dq=dq,root=root,gyro=gyro)
                    if event is not None:
                        write(dict(event,actions_sent=sent_count))
                        print('TELEOP_MODE: '+str(event),flush=True)
                        if event['record_type']=='teleop_mode_requested':request='toggle'
                    if not tracking and modes.pending is not None:
                        try:
                            candidate,diag=manual_reference(runtime,snapshot,clock.monotonic())
                            if diag['pico_age_s']>runtime.config['pico_hold_age_s']:candidate=None
                        except (ValueError,TimeoutError):
                            candidate=None;modes.cancel()
                        if modes.ready(snapshot,clock.monotonic(),q=q,dq=dq,root=root,gyro=gyro,
                                preparation_done=elapsed>=preparation,motion=candidate):
                            request='track'
                            print('TELEOP_MODE: '+modes.mode+'; reference handover, then center stick to arm.',flush=True)
                            write(dict(record_type='teleop_mode_selected',mode=modes.mode,actions_sent=sent_count))
                from .actor_teleop_modes import active_planner
                planner = active_planner(runtime)
                if planner is not None and tracking:
                    if request == 'toggle' and planner.stop_started is None:
                        planner.stop_started = clock.monotonic()
                        write(dict(record_type='planner_stop_requested', actions_sent=sent_count))
                    if planner.stop_started is not None:
                        # Let the generated gait decelerate before returning to stand.
                        legs = [i for i,n in enumerate(runtime.model.contract.names)
                                if any(k in n for k in ('hip','knee','ankle'))]
                        settled = (clock.monotonic()-planner.stop_started >= 2.
                                   and np.max(np.abs(dq[legs])) < .4)
                        if modes is not None:
                            quiet=(np.max(np.abs(dq[legs]))<.15 and np.max(np.abs(gyro))<.1
                                and planner.future is None and planner.clip is not None
                                and planner.cursor>=planner.origin+len(planner.clip[0])-1)
                            modes.quiet_since=(clock.monotonic() if modes.quiet_since is None else modes.quiet_since) if quiet else None
                            settled=settled and modes.quiet_since is not None and clock.monotonic()-modes.quiet_since>=.5
                        request = 'toggle' if settled else None
                if request == 'toggle' and tracking:
                    # An intentional standby transition; a source fault takes
                    # handle_pico_loss (pico_loss_action) instead of this path.
                    if planner is None:
                        manual_reference(runtime,snapshot,clock.monotonic())
                    tracking = False
                    auto_pico = False
                    transition_q=q if modes is None or modes.last_q is None else modes.last_q
                    transition_root=root if modes is None or modes.last_root is None else modes.last_root
                    stand_target=runtime.hold_motion()
                    if modes is not None and modes.pending is not None:
                        # R3 returns only the lower body to stand. Arm targets
                        # continue from Pico below, using the same filter state.
                        stand_target=runtime.math.Motion(stand_target.q.copy(),stand_target.base.copy(),stand_target.quat.copy())
                        slots=modes.upper_follow.slots
                        stand_target.q[:,slots]=transition_q[slots]
                    lead = stand_target.lead_in(
                        transition_q,transition_root,runtime.config['body_profile']['lead_in'])
                    offset = runtime.math.yaw(transition_root)
                    if modes is not None:modes.paused()
                    stand_tick = 0
                    elapsed = reference_time = 0.
                    reference_frame = 0
                    preparation = max(.3,(len(lead.q)-1)*runtime.model.contract.period)
                    request = None
                    print('PICO_STANDBY: returning reference to ONNX stand; EXTERNAL session stays active.',flush=True)
                    write(dict(record_type='tracking_paused',actions_sent=sent_count,
                               transition_start_q=transition_q.tolist(),policy_history_reset=False))
                automatic_switch = False
                if auto_pico and not tracking and elapsed >= preparation and snapshot.get('state') == 'ready':
                    try:
                        manual_reference(runtime,snapshot,clock.monotonic())
                        automatic_switch = True
                    except (ValueError,TimeoutError):
                        pass
                if (request in ('track','toggle') or automatic_switch) and not tracking:
                    try:
                        if elapsed < preparation:
                            raise ValueError('Stand preparation has not completed')
                        motion, _ = manual_reference(runtime, snapshot, clock.monotonic())
                    except (ValueError, TimeoutError) as error:
                        print(f'PICO_SWITCH_REJECTED: {error}; ONNX stand continues. No switch is queued.', flush=True)
                        write(dict(record_type='pico_switch_rejected', error=str(error), actions_sent=sent_count))
                    else:
                        tracking = True
                        # GMR keeps its world root. Align heading at each new
                        # engagement using the current measured robot heading.
                        tracking_yaw = reference_yaw(runtime, motion, root)
                        if modes is not None:
                            modes.engage(lead.q[-1],runtime.math.multiply(runtime.math.yaw_quaternion(offset),lead.quat[-1]))
                        print('PICO_TRACKING: same EXTERNAL session and policy history.', flush=True)
                        write(dict(record_type='tracking_started',actions_sent=sent_count,
                                   reference_yaw_offset=tracking_yaw))
            published_action = runtime.policy.last_action.copy()
            try:
                if motion_source is not None and elapsed >= preparation:
                    if not tracking:
                        motion_source.begin(q, root)
                        tracking = True
                        write(dict(record_type='motion_started', actions_sent=sent_count,
                            motion=motion_source.metadata, lead_frames=motion_source.lead_frames,
                            total_ticks=motion_source.total_ticks, policy_history_reset=False))
                        print('MOTION_PLAYING: one pass, live robot feedback; STOP/Ctrl-C cancels.',flush=True)
                    if motion_source.finished:
                        motion_complete = True
                        break
                    reference_frame = motion_source.tick
                    reference_time = reference_frame*runtime.model.contract.period
                    command, obs, row = motion_source.infer(q=q, dq=dq, root=root, gyro=gyro)
                elif manual and tracking:
                    command, obs, row = pico_reference_command(runtime, snapshot, clock.monotonic(),
                        q=q, dq=dq, root=root, gyro=gyro, yaw_offset=tracking_yaw)
                elif requires_pico and not pico_reengage_blocked and elapsed >= preparation:
                    if not tracking:
                        if not inputs.reference_ready():
                            raise TimeoutError('Pico reference window not ready after preparation')
                        motion, _ = manual_reference(runtime, inputs.snapshot, clock.monotonic())
                        tracking_yaw = reference_yaw(runtime, motion, root)
                        tracking = True
                        write(dict(record_type='tracking_started', actions_sent=sent_count,
                                   reference_yaw_offset=tracking_yaw))
                    command, obs, row = pico_reference_command(runtime, inputs.snapshot, clock.monotonic(),
                        q=q, dq=dq, root=root, gyro=gyro, yaw_offset=tracking_yaw)
                else:
                    stand_motion=lead
                    if modes is not None and modes.upper_follow is not None:
                        try:
                            upper,diag=manual_reference(runtime,snapshot,clock.monotonic())
                            upper_q=upper.q if diag['pico_age_s']<=runtime.config['pico_hold_age_s'] else None
                        except (ValueError,TimeoutError):upper_q=None
                        # Infer a current-frame window so the persistent arm
                        # filter is evaluated at frame zero, not a lead-in offset.
                        start=min(round(reference_time/runtime.model.contract.period),len(lead.q)-1)
                        indices=np.minimum(start+np.arange(runtime.reference.steps),len(lead.q)-1)
                        stand_motion=runtime.math.Motion(lead.q[indices],lead.base[indices],lead.quat[indices])
                        stand_motion=modes.follow_upper(stand_motion,upper_q)
                    command, obs, row = runtime.infer_motion_command(stand_motion, q=q, dq=dq,
                        root=root, gyro=gyro, ref_time=0. if stand_motion is not lead else reference_time, yaw_offset=offset)
            except (ValueError, TimeoutError) as error:
                if not handle_pico_loss(error, q, root):
                    raise
                deadline = advance(deadline)
                continue
            last_inferred_identity = identity
            last_inferred_receipt_ns = int(state.received_monotonic_ns)
            # SDK decode updates last_action. Commit that candidate only after
            # publication succeeds; skipped ticks keep the last published action.
            pending_action = runtime.policy.last_action.copy()
            runtime.policy.last_action = published_action
            if modes is not None:
                row['upper_body_source']='pico' if modes.upper_follow is not None else 'stand'
            command, pending_head = runtime.apply_head(command, row, now=clock.monotonic(),
                measured_q=state.q, tracking=(tracking or (modes is not None and modes.upper_follow is not None)) and motion_source is None and (
                    getattr(runtime, 'planner', None) is None or (snapshot is not None and
                    ('head_snapshot' in snapshot or 'received_monotonic' in snapshot))),
                snapshot=(snapshot if manual else inputs.snapshot) if (tracking or (modes is not None and modes.upper_follow is not None)) else None)
            envelope.check(command)
            compute_end = clock.monotonic()
            record_period_overrun(write,'compute',compute_end-cycle,sent_count,
                status_ms=(status_end-cycle)*1000,state_ms=(state_end-status_end)*1000,
                policy_ms=row.get('total_ms'),
                thread_cpu_ms=(cpu_clock()-cpu_start)*1000 if cpu_start is not None else None)
            try:
                inputs.fresh(state)
            except StaleStateError as error:
                # Keep the measured observation; the pending action is not committed.
                skip_stale('before_publish', error, state_id=int(state.state_id),
                    at_read=state_age_at_read, at_failure=inputs.ages(state),
                    compute_ms=(compute_end-cycle)*1000, policy_ms=row.get('total_ms'))
                deadline = advance(deadline)
                continue
            if requires_pico and not pico_reengage_blocked:
                try:
                    inputs.require_pico()
                except TimeoutError as error:
                    if not handle_pico_loss(error, q, root):
                        raise
                    deadline = advance(deadline)
                    continue
            if manual and tracking and active_planner(runtime) is None:
                # Recheck source age immediately before publishing. On loss,
                # handle_pico_loss returns to stand without auto reconnect.
                age = clock.monotonic()-float(snapshot['received_monotonic'])
                if not 0 <= age <= pico_max_age:
                    error = TimeoutError('Pico reference expired before publication')
                    if not handle_pico_loss(error, q, root):
                        raise error
                    deadline = advance(deadline)
                    continue
            guard.check(controller.status, entered)
            if stop.is_set():
                break
            command = replace(command, input_state_id=int(state.state_id),
                input_state_ts_real_us=int(state.ts_state_real_us),
                ts_client_created_us=clock.time_ns()//1000)
            state_age_before_publish = inputs.ages(state)
            before = clock.monotonic()
            # SDK alone owns session + sequence, shared with end-effector output.
            sent = controller.publish(command, target_names=envelope.target_names,
                                      required_fields=BODY_FIELDS_ALL)
            runtime.policy.last_action = pending_action
            if runtime.head is not None:
                runtime.head.commit(pending_head)
            sent_count += 1
            stale_ticks = 0
            policy_tick += 1
            if not tracking:
                stand_tick += 1
            elif motion_source is not None:
                motion_source.advance()
            report['actions_sent'] = sent_count
            if auto and not report['onnx_control_enabled']:
                print('ONNX_READY: policy commands started; '+('NPZ playback follows stand preparation'
                      if motion_source is not None else 'both Pico triggers toggle following'),flush=True)
            report['onnx_control_enabled'] = True
            write(dict(row, actions_sent=sent_count, control_handles_created=1,
                stage=('motion_tracking' if motion_source is not None else 'pico_tracking') if tracking else 'prepare_stand',
                policy_tick=policy_tick-1,
                reference_time_s=0. if tracking and motion_source is None else reference_time,
                reference_frame=0 if tracking and motion_source is None else reference_frame,
                state_id=int(state.state_id), source_state_us=int(state.ts_state_real_us),
                feedback_timing=dict(at_read=state_age_at_read, before_publish=state_age_before_publish,
                    publication_us=int(state.ts_state_pub_us),
                    received_monotonic_ns=int(state.received_monotonic_ns),
                    pull_context_receive_stamp_ns=int(state.pull_context_receive_stamp_ns)),
                feedback=dict(q_body=state.q.tolist(),dq_body=state.dq.tolist(),
                    root_wxyz=root.tolist(),gyro=gyro.tolist(),
                    applied_session_id=int(state.applied_session_id),
                    applied_sequence=int(state.applied_sequence),
                    ts_first_output_us=int(state.ts_first_output_us)),
                session_id=sent.session_id, sequence=sent.sequence,
                publish_ms=(clock.monotonic()-before)*1000,
                compute_ms=(compute_end-cycle)*1000,
                sent_body_targets=[dict(q=j.q, dq=j.dq, tau=j.tau, kp=j.kp, kd=j.kd) for j in sent.joints]), obs)
            if sent_count % 50 == 0:
                stage = ('motion_tracking' if motion_source is not None else 'pico_tracking') if tracking else 'stand'
                print(f'ACTOR_LIVE steps={sent_count} stage={stage}', flush=True)
            complete = clock.monotonic()
            record_period_overrun(write,'complete_cycle',complete-cycle,sent_count)
            deadline = advance(deadline)
        report['stopped_by_signal'] = stop.is_set()
        report['elapsed_s'] = clock.monotonic()-start
        report['stop_reason'] = 'operator_stop' if stop.is_set() else 'motion_complete' if motion_complete else 'duration_elapsed'
        if motion_source is not None:
            report['motion'] = motion_source.metadata
            report['motion_ticks'] = motion_source.tick
        write(dict(record_type='stop_requested',reason=report['stop_reason'],
            elapsed_s=report['elapsed_s'],duration_limit_s=args.seconds,actions_sent=sent_count))
        print(f"ACTOR_STOP: {report['stop_reason']}, elapsed={report['elapsed_s']:.2f}s; CANCEL -> PASSIVE",flush=True)
        return report
    finally:
        if grippers is not None:
            # Opens and releases EE clients while EXTERNAL is still owned.
            try:
                grippers.close()
                write_gripper_events()
            except BaseException as error:
                print(f'PICO_GRIPPER close: {error}', file=sys.stderr, flush=True)
        if auto and not attempted:
            cancel_startup(controller, cancel_action, clock, write, startup_generation)
        if attempted:
            original = sys.exc_info()[1]
            try:
                result = controller.exit_external(timeout=5.)
                write(dict(record_type='external_exit', status=status_row(result), actions_sent=sent_count))
                if result.current_action != 'PASSIVE' or phase(result) != 'IDLE':
                    raise RuntimeError('CANCEL did not confirm PASSIVE/IDLE')
                report['exit_confirmed'] = True
            except BaseException as error:
                print(f'EXIT INCOMPLETE: {error}. Inspect actual state; use hardware E-stop if safety is uncertain.',
                      file=sys.stderr, flush=True)
                # Preserve the original fault while recording exit failure too.
                try:
                    write(dict(record_type='exit_failure', error=repr(error),
                               actual_status=status_row(controller.status), actions_sent=sent_count))
                except BaseException as log_error:
                    print(f'Cannot record exit failure: {log_error}', file=sys.stderr, flush=True)
                if original is None:
                    raise
