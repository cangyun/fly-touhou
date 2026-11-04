"""Between-rollout learning rates; optional progress schedule independent of KL."""
import math


class PolicyLearningRate:
    def __init__(self,optimizer,*,scale=None,adaptive=False,target_kl=.03,min_scale=None,kl_early_stop=True):
        policy_groups=[g for g in optimizer.param_groups if g['name']!='three_critics']
        if min_scale is None:min_scale=policy_groups[0].get('policy_min_lr_scale',.05)
        if not math.isfinite(min_scale) or not 0<min_scale<=1:
            raise ValueError('Positive finite minimum policy LR multiplier at most1 required')
        for group in policy_groups:
            group.setdefault('policy_base_lr',group['lr'])
            group.setdefault('policy_group_multiplier',1.)
            group.setdefault('policy_reference_base_lr',group['policy_base_lr']/group['policy_group_multiplier'])
        for group in policy_groups:group['policy_min_lr_scale']=min_scale
        saved=policy_groups[0].get('policy_progress_schedule')
        self.progress_schedule=self.validate_progress_schedule(saved)
        if scale is None:scale=policy_groups[0]['lr']/(policy_groups[0]['policy_base_lr']*self.progress_factor())
        if not min_scale<=scale<=1:
            raise ValueError('Policy LR scale within its minimum and1 required')
        self.optimizer=optimizer;self.scale=scale;self.min_scale=min_scale
        self.configure_kl(adaptive=adaptive,kl_early_stop=kl_early_stop,target_kl=target_kl)
        self.apply()
    def configure_kl(self,*,adaptive,kl_early_stop,target_kl):
        if type(adaptive) is not bool or type(kl_early_stop) is not bool:
            raise ValueError('Boolean KL stopping and adaptive LR settings required')
        if adaptive or kl_early_stop:
            if (isinstance(target_kl,bool) or not isinstance(target_kl,(int,float))
                    or not math.isfinite(target_kl) or target_kl<=0):
                raise ValueError('Enabled KL control needs a positive finite target')
        else:target_kl=None
        if adaptive and getattr(self,'progress_schedule',None) is not None:
            raise ValueError('Progress scheduling does not combine with KL-adaptive rates')
        self.adaptive=adaptive;self.kl_early_stop=kl_early_stop;self.target=target_kl
    @staticmethod
    def validate_progress_schedule(schedule):
        if schedule is None:return None
        if not isinstance(schedule,dict) or set(schedule)!={'kind','hold_updates','decay_updates','minimum_factor','completed_updates'}:
            raise ValueError('A complete progress learning-rate schedule is required')
        if schedule['kind']!='hold_cosine':raise ValueError('Expected hold_cosine schedule')
        for key,minimum in (('hold_updates',0),('decay_updates',1),('completed_updates',0)):
            if type(schedule[key]) is not int or schedule[key]<minimum:raise ValueError('Invalid schedule update count')
        factor=schedule['minimum_factor']
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 0<factor<=1:
            raise ValueError('Schedule minimum factor must be in(0,1]')
        return dict(schedule)
    def progress_factor(self):
        schedule=getattr(self,'progress_schedule',None)
        if schedule is None:return 1.
        fraction=min(1.,max(0.,(schedule['completed_updates']-schedule['hold_updates'])/schedule['decay_updates']))
        return schedule['minimum_factor']+(1.-schedule['minimum_factor'])*.5*(1.+math.cos(math.pi*fraction))
    def configure_progress(self,schedule):
        checked=self.validate_progress_schedule(schedule)
        if checked is not None and self.adaptive:raise ValueError('Progress scheduling must be independent of KL')
        self.progress_schedule=checked;self.apply()
    def apply(self):
        for group in self.optimizer.param_groups:
            if group['name']!='three_critics':
                group['lr']=group['policy_base_lr']*self.scale*self.progress_factor()
                schedule=getattr(self,'progress_schedule',None)
                if schedule is None:group.pop('policy_progress_schedule',None)
                else:group['policy_progress_schedule']=dict(schedule)
    def validate_group_multiplier(self,name,multiplier):
        if (isinstance(multiplier,bool) or not isinstance(multiplier,(int,float))
                or not math.isfinite(multiplier) or multiplier<=0):
            raise ValueError('Positive finite policy-group multiplier required')
        groups=[group for group in self.optimizer.param_groups if group['name']==name and name!='three_critics']
        if len(groups)!=1:raise ValueError('One named policy group required')
        group=groups[0]
        if not math.isfinite(group['policy_reference_base_lr']*multiplier):
            raise ValueError('Finite policy-group base rate required')
        return group
    def set_group_multipliers(self,multipliers):
        """Change relative rates together, without resetting any Adam moments.

        Readouts and the spiking core can use different relative rates. All
        policy groups share the configured scale and optional progress schedule.
        Changing group multipliers does not enable KL stopping or KL feedback.
        """
        if not isinstance(multipliers,dict):raise ValueError('Named policy-group multiplier mapping required')
        for name,multiplier in multipliers.items():self.validate_group_multiplier(name,multiplier)
        return [self.set_group_multiplier(name,multiplier) for name,multiplier in multipliers.items()]
    def set_group_multiplier(self,name,multiplier):
        """Set one policy group's LR relative to its original base, keeping Adam state."""
        group=self.validate_group_multiplier(name,multiplier)
        before=group['policy_group_multiplier']
        group['policy_group_multiplier']=float(multiplier)
        group['policy_base_lr']=group['policy_reference_base_lr']*multiplier
        self.apply()
        return dict(group=name,before=before,after=float(multiplier),
            effective_lr=group['lr'],optimizer_moments_preserved=True)
    def after_rollout(self,logs,*,early_stop=None,growth_fraction=1.):
        if not 0<growth_fraction<=1:raise ValueError('LR growth fraction must be in(0,1]')
        rows=sum(r.get('trained_student_rows',1) for r in logs)
        kl=(sum(r.get('exact_action_KL',r['KL'])*r.get('trained_student_rows',1) for r in logs)/rows
            if rows else 0.)
        # Include the measured stop condition; otherwise a shortened low-KL
        # prefix would wrongly increase LR immediately after a KL stop.
        feedback=max(kl,float(early_stop['measured_KL'])) if early_stop else kl
        if not math.isfinite(kl):raise FloatingPointError('Nonfinite measured KL')
        before=self.scale;factor=1.;progress_before=self.progress_factor()
        if self.adaptive:
            if feedback>self.target:factor=max(.25,math.sqrt(self.target/feedback))
            elif feedback<.5*self.target:factor=1.25**growth_fraction
            self.scale=min(1.,max(self.min_scale,self.scale*factor));self.apply()
        if getattr(self,'progress_schedule',None) is not None:
            # Count actual completed Adam updates, not requested epochs,
            # collected frames, KL magnitude, or rejected/unused samples.
            self.progress_schedule['completed_updates']+=len(logs)
            self.apply()
        return dict(measured_rollout_mean_KL=kl,feedback_KL=feedback,early_stop=early_stop,target_KL=self.target,
            adaptive_policy_lr=self.adaptive,KL_early_stopping_enabled=self.kl_early_stop,
            progress_lr_schedule=dict(self.progress_schedule) if getattr(self,'progress_schedule',None) else None,
            progress_lr_factor_before=progress_before,progress_lr_factor_after=self.progress_factor(),
            low_KL_growth_fraction=growth_fraction,
            policy_lr_scale_before=before,policy_lr_scale_after=self.scale,
            policy_min_lr_scale=self.min_scale,
            policy_lr_floor_limited=bool(self.adaptive and feedback>self.target and self.scale<=self.min_scale),
            learning_rates={g['name']:g['lr'] for g in self.optimizer.param_groups},
            policy_group_multipliers={g['name']:g['policy_group_multiplier'] for g in self.optimizer.param_groups
                if g['name']!='three_critics'},
            applied_optimizer_updates_kept=len(logs),updates_rejected=0,training_worlds_reset=False)
