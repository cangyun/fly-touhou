"""Read-only reward/terminal-route review of the completed original-game round."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
FROZEN=Path('./outputs/source_snapshots/20261025-B1-BC-only-full-story-development-v2')
sys.path.insert(0,str(FROZEN/'src'))
from fly_touhou.thrl_durable_records import write_json as write
from fly_touhou.thrl_study_io import read_shared_json as read,utc_now


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probes(receipt):
    import numpy as np
    from fly_touhou.thrl_movement_teacher import MovementSafetyTeacher,normal_unprotected
    t=read(receipt)
    if digest(t['arrays'])!=t['sha256']:raise ValueError('Recorded terminal arrays changed')
    teacher=MovementSafetyTeacher()
    with np.load(t['arrays'],allow_pickle=False) as z:
        f,o,actions=z['features'],z['objects'],z['action']
        if not np.array_equal(actions,np.argmax(z['probabilities'],axis=1)):
            raise ValueError('These recorded actions were not the measured greedy decisions')
        ids=np.flatnonzero(normal_unprotected(f))
        selected=sorted(set(int(ids[max(0,len(ids)-1-offset)]) for offset in (0,3,6))) if len(ids) else []
        results=[]
        for i in selected:
            feature,objects=f[i],o[i];maps=z['maps'][i]
            origin,*_=teacher._scene(feature,objects)
            unknown={str(h):teacher._uncertainty(feature,objects,maps,horizon=h) for h in (4,12,24)}
            clear=teacher.clearances(feature,objects,np.repeat(origin[None],18,axis=0),teacher.velocities,duration=4)
            mask,route=teacher.label(feature,objects,maps,force=True)
            action=int(actions[i]);present=objects[:,0]>.5
            results.append(dict(row=i,episode_frame=int(z['episode_frame'][i]),
                stage_frame=int(z['stage_frame'][i]),action=action,
                native_player_circle_component=float(feature[13]*64),
                player_rectangle_half_extents=(feature[14:16]*64).astype(float).tolist(),
                Bomb_inventory=float(feature[4]*8),position=origin.tolist(),
                selected_objects=int(present.sum()),point_hazards=int((present&((objects[:,1]>.5)|(objects[:,2]>.5))&(objects[:,13]>.5)).sum()),
                laser_objects=int((present&((objects[:,3]>.5)|(objects[:,4]>.5))).sum()),
                line_laser_objects=int((present&(objects[:,3]>.5)).sum()),
                curve_laser_segments=int((present&(objects[:,4]>.5)).sum()),
                laser_motion_observed=int((present&((objects[:,3]>.5)|(objects[:,4]>.5))&(objects[:,26]>.5)).sum()),
                uncertainty_by_horizon=unknown,
                modeled_chosen4_clearance=float(clear[action]) if action<18 and np.isfinite(clear[action]) else None,
                modeled_best4_clearance=float(clear.max()) if np.isfinite(clear.max()) else None,
                modeled_other_safe4_actions=np.flatnonzero(clear>1).tolist(),
                modeled_directional_mistake_candidate=bool(action<18 and unknown['4'] is None
                    and clear[action]<=0 and clear.max()>1),
                modeled_route_reason=route['reason'],modeled_safe24_first_actions=np.flatnonzero(mask).tolist(),
                alternative_actions_executed_in_native_game=False))
        last_index=int(ids[-1]) if len(ids) else len(f)-1
        return dict(history_rows=len(f),normal_unprotected_rows=len(ids),
            last_normal_player_radius=float(f[last_index,13]*64),
            final_observed_player_radius=float(f[-1,13]*64),probes=results,
            arrays=t['arrays'],arrays_sha256=t['sha256'])


def main():
    r=ROOT/'artifacts/engineering/B1-autonomous-dodge-20261024'
    plan=read(r/'BC-only-full-story-development20-v2-control/plan.json')
    base=Path(plan['after_output'])
    if digest(base/'report.json')!=plan['after_report_sha256']:raise ValueError('Completed fixed140 changed')
    a=read(base/'full_story_initial/report.json');c=read(base/'full_story_final/report.json')
    tail=read(r/'current-final13-GameOver-measured-deathbomb-tail-audit-v1.json')
    accounting=read(r/'frozen40-life-accounting-and-shaped-return-v1.json')
    cfg=read(plan['environment_config']);gamma=cfg['parameters']['gamma'];lam=cfg['parameters']['gae_lambda']
    bc=read(r/'BC-only-full-story-development20-completed-evidence-v1.json') if (r/'BC-only-full-story-development20-completed-evidence-v1.json').exists() else None
    neural=read(r/'final13-exact-policy-neural-risk-decomposition-v1.json') if (r/'final13-exact-policy-neural-risk-decomposition-v1.json').exists() else None
    calibration=read(r/'final-point-risk-player-geometry-calibration-scope-v1.json') if (r/'final-point-risk-player-geometry-calibration-scope-v1.json').exists() else None
    demo=read(r/'real-neural-visible-clear-completed-evidence-v1.json') if (r/'real-neural-visible-clear-completed-evidence-v1.json').exists() else None
    cases=[]
    lookup={row['evaluation_game']:row for row in c['episode_results']}
    for original in tail['cases']:
        row=lookup[original['game']]
        if row['outcome']!='gameover' or row['native_seed']!=original['native_seed']:
            raise ValueError('Failure case does not match this exact final neural policy')
        probe=probes(original['terminal_trace']);missed=original['terminal_tail_missed_usable_Bomb_decisions']
        note=('末次中弹已读到有库存的决死窗口，却没有选 Bomb；是否能救活尚未执行反事实。'
            if missed else '保存的终局窗口没有发现有库存且可用的漏决死；最后库存为0，需追查此前路线与资源消耗。')
        cases.append({**original,'terminal_probe':probe,'evidence_conclusion':note,
            'observed_collision_events':row['movement_diagnostics']['observed_collision_events'],
            'confirmed_deathbombs':row['reward_diagnostics']['confirmed_deathbombs'],
            'point_items':row['point_items'],'terminal_stage_frame':row['stage_frame'],
            'undiscounted_shaped_reward_sum':sum(row['reward_vector'])})
    result=dict(schema='B1-stable-clear-reward-and-final13-review-v1',utc=utc_now(),
        analysis_only=True,reward_modified=False,new_optimizer_updates=0,
        formal_comparison=dict(ancestor_clears=a['clears'],final_clears=c['clears'],episodes_each=20,
            ancestor_mean_stage=a['mean_max_stage'],final_mean_stage=c['mean_max_stage']),
        completed_BC_full_story_development20=bc,
        exact_policy_risk_decomposition=neural,original_risk_calibration_scope=calibration,
        completed_real_neural_visible_demonstration=demo,
        actual_discount=dict(gamma=gamma,gae_lambda=lam,
            reward_half_life_decisions=__import__('math').log(.5)/__import__('math').log(gamma),
            GAE_e_folding_decisions=-1/__import__('math').log(gamma*lam),
            final_clear_lengths=[x['evaluation_decisions'] for x in c['episode_results'] if x['outcome']=='clear'],
            final_clear_mean_opening_weight=sum(gamma**(x['evaluation_decisions']-1)
                for x in c['episode_results'] if x['outcome']=='clear')/c['clears']),
        observed_failure_stages=dict(Counter(row['stage'] for row in cases)),cases=cases,
        prior_life_accounting=str(r/'frozen40-life-accounting-and-shaped-return-v1.json'),
        sources=[dict(path=str(p),sha256=digest(p)) for p in [
            base/'report.json',base/'full_story_initial/report.json',base/'full_story_final/report.json',
            r/'current-final13-GameOver-measured-deathbomb-tail-audit-v1.json',
            FROZEN/'src/fly_touhou/th14_completion_v3.py',
            FROZEN/'src/fly_touhou/thrl_movement_teacher.py']],
        scope='Recorded native outcomes and bounded saved tails; modeled routes are hypotheses, not native counterfactuals')
    target=r/'stable-clear-wrapup-review-v1.json';write(target,result)
    names={'no_feasible_first_action':'当下已无模型内安全首步','all_first_actions_feasible':'短程模型未区分出方向',
        'no_validated_two_phase_route':'未找到24帧两段安全路线','directional_route':'存在模型内24帧路线',
        'clipped_collision_geometry':'近处几何发生裁剪','laser_motion_unknown':'激光运动不确定',
        'nearby_unrepresented_map_hazard':'附近地图威胁未被对象覆盖','growth_motion_unknown':'变大中的弹幕运动不确定',
        'unrepresented_laser_map_motion_unknown':'地图中另有运动未知的激光'}
    table=[]
    for row in cases:
        ps=row['terminal_probe']['probes'];last=ps[-1] if ps else {}
        evidence='漏决死（有库存）' if row['terminal_tail_missed_usable_Bomb_decisions'] else '终局无 Bomb'
        details=html.escape(row['evidence_conclusion'])
        if last:
            unknown=last['uncertainty_by_horizon']['4']
            details+=f"<br>末次正常状态：{last['episode_frame']}帧；对象{last['selected_objects']}/96；人物圆判定分量{last['native_player_circle_component']:.2f}。"
            details+='<br>短程几何：'+html.escape(names.get(unknown or last['modeled_route_reason'],unknown or last['modeled_route_reason']))+'。'
            if last['modeled_directional_mistake_candidate']:
                details+='<br>模型预测本次选择会碰撞，另有正余量动作；须原作反事实验证后才可认定选错。'
        for miss in row['terminal_tail_missed_usable_Bomb_decisions']:
            details+=(f"<br>结束前{miss['frames_to_terminal']}原作帧：Bomb库存{miss['Bomb_inventory_encoded']*8:.0f}，"
                f"剩余决死{miss['remaining_native_frames_encoded']:.0f}帧；选择移动，"
                f"Bomb概率{miss['Bomb_probability']*100:.3f}%，所选动作概率{miss['largest_action_probability']*100:.1f}%。")
        table.append(f"<tr><td>{row['game']}</td><td>{row['native_seed']}</td><td>{row['stage']}</td>"
            f"<td>{row['native_frames']:,}</td><td>{row['Miss']}</td><td>{row['Bomb']}</td>"
            f"<td>{row['net_life_increases']}</td><td>{evidence}</td><td><details><summary>查看证据</summary>{details}</details></td></tr>")
    def mean(rows,fn):return sum(fn(x) for x in rows)/len(rows)
    def life(x):return x['endpoints']['end']['lives']-x['endpoints']['start']['lives']+x['misses']
    av=a['episode_results'];cv=c['episode_results']
    checks=[('通关',f"{a['clears']}/20",f"{c['clears']}/20"),
        ('平均推进面数',f"{a['mean_max_stage']:.2f}",f"{c['mean_max_stage']:.2f}"),
        ('平均原作帧',f"{a['mean_native_frames']:,.0f}",f"{c['mean_native_frames']:,.0f}"),
        ('平均 Miss',f"{a['mean_misses']:.2f}",f"{c['mean_misses']:.2f}"),
        ('平均 Bomb',f"{a['mean_bombs_used']:.2f}",f"{c['mean_bombs_used']:.2f}"),
        ('平均净增残机',f"{mean(av,life):.2f}",f"{mean(cv,life):.2f}"),
        ('平均累计奖励（未折扣）',f"{mean(av,lambda x:sum(x['reward_vector'])):.2f}",f"{mean(cv,lambda x:sum(x['reward_vector'])):.2f}")]
    compare=''.join('<tr>'+''.join(f'<td>{html.escape(str(t))}</td>' for t in row)+'</tr>' for row in checks)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B1通关、奖励与13局失败复盘</title>
<style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f8;color:#172534;margin:0}main{max-width:1200px;margin:30px auto;padding:32px;background:white;border-radius:14px}h1{font-size:27px}h2{font-size:21px;margin-top:35px}table{border-collapse:collapse;width:100%;margin:15px 0;font-size:14px}th,td{border-bottom:1px solid #dfe5ee;padding:9px 10px;text-align:left;vertical-align:top}th{background:#edf3fa}p,li{max-width:1120px}strong{color:#123f77}.note{padding:15px 20px;background:#edf4fb;border-left:4px solid #216aa6}summary{cursor:pointer;color:#175da4}details{max-width:480px}.small{font-size:13px;color:#546474}code{font-family:Consolas,monospace}</style><main>
<h1>B1 通关、奖励与13局失败复盘</h1><p class="note">已证明单个神经网络可以完成原作 Lunatic 六面；当前固定测试仍为7/20，尚未达到稳定通关。本文件只分析，不修改奖励、PPO、模型或训练预算。</p>
<h2>一、这轮究竟改善了什么</h2><table><tr><th>同一组20个初始种子</th><th>原B1</th><th>完整教师+PPO</th></tr>'''+compare+'''</table>
<p>两模型仅1个种子共同通关；旧模型独有6局、新模型独有6局。通关率差为0，McNemar精确检验p=1；每组35%的Wilson 95%区间约18.1%–56.7%。样本不足以宣称相当或改进。匹配的是初始原作RNG，不是之后完全相同的弹幕：动作不同会改变后续状态。</p>
<p>更少的 Miss、Bomb 同时伴随更早结束和更少净增残机，不能独立证明避弹更好。禁Bomb评估中，模型存活明显长于随机移动（描述性比较），说明已有避弹能力；新旧神经模型之间尚无明确提升证据。普通Bomb大幅减少，也不等于路线更安全。</p>
<h2>二、奖励机制的判断</h2>
<ul><li><strong>保留的合理部分：</strong>真实 Miss −2、真实Game Over额外−2、真实六面通关+5；只奖励可受伤时段的存活，不奖励无敌、对话或复活拖时间；Boss每阶段伤害奖励最多0.2；不凭消失物品猜拾取，不把规则教师直接变成执行器。</li>
<li><strong>长程目标偏弱：</strong>实际γ=0.999，每决策4原作帧；奖励半衰期约693次决策，约46秒原作时间。通关需约2.5万次决策，开局看到的通关奖励权重接近10⁻¹¹。GAE的指数衰减尺度约167次决策；价值自举仍能传递更长反馈，并非模型只能学习这个长度。72局完整的神经网络训练轨迹没有通关，稀疏终点信号还缺少正样本。</li>
<li><strong>节省Bomb和避弹混在一起：</strong>普通/未确认Bomb −0.5，确认决死 −0.1，而被决死救回的碰撞没有独立碰撞惩罚。这会使“先撞再决死”比提前普通Bomb便宜0.4；不能据此断言网络故意利用漏洞，但目标确实不直接等价于少碰撞或稳定通关。</li>
<li><strong>额外生命缺少直接目标：</strong>没有拾点、残机增长的直接正奖励。多得生命意味着多一次续命机会，但也可能多一次−2的Miss；若远期通关收益折扣很小，省掉奖励中的死亡损失未必提升通关。新模型平均净增残机4.90→3.35，必须把资源获取和生存分开审查，不能只按死亡数排名。</li>
<li><strong>火力势函数不是最终资源目标：</strong>γΦ(s′)−Φ(s)，Φ=0.2×P/400，且真实终局Φ=0。在相同γ、完整折扣回报和理想求解下会望远镜相消为−Φ(s₀)，只提供局部学习提示；它本身不改变最优策略去“多拿P”，也不奖励净增残机。未折扣累计值受长度影响，不能据此给策略定优劣。</li>
<li><strong>实际出现了目标不一致的迹象：</strong>平均未折扣累计奖励1.42→6.36，固定测试通关仍7/20。这个值不是PPO优化的折扣回报或优势函数，不能称为奖励投机的证明；但足以要求停止用总奖励代替通关率验收。</li></ul>
<h2>三、13局失败逐局记录</h2><p>5局止于第四面，8局止于第六面。下表的残机是整局净增库存，不代表逐件拾取计数；“漏决死”只使用已经读到的库存、状态4和剩余≥2帧的窗口。没有执行替代动作，因此不声称某一次Bomb必能救活。</p>
<table><tr><th>局号</th><th>原作种子</th><th>终止面</th><th>原作帧</th><th>Miss</th><th>Bomb</th><th>净增残机</th><th>直接证据</th><th>尾段诊断</th></tr>'''+''.join(table)+'''</table>
<p>第5、15、16局在终局前有库存、有足够观测决死时间，最高概率动作仍不是Bomb。其中第16局一次Bomb概率约0.2708，最高动作约0.2739，差距很小；第5局Bomb概率约0.00060，属于明显置信不足。这3局的玩家圆判定分量均为21.6，尺寸已经进入输入，并被点风险几何计算使用。需要调查巨大化关键状态的训练覆盖与泛化，不能再简单归因于“没读到人物尺寸”。</p>
<p>其余10局在保存的终局尾段未发现有库存的漏决死，最后库存为0。这只是终局事实；不能推断整局资源使用已经最优。每面只保存首次观察碰撞和终局前128次决策，覆盖并不完整，不能根据这些短尾段准确确定更早是哪一步选入了死路。几何分析仅是当前速度外推的4/12/24帧假设，未知激光、变大弹幕或漏覆盖时必须弃权。</p>
<p><strong>进一步的尾段证据：</strong>8局第六面失败中，7局最后正常状态的人物圆判定分量为21.6，另1局为2.0；第4、9、15、16局的末次正常状态，在通过4帧不确定性检查的几何模型中存在其他正余量动作，而实际选择的动作预测碰撞，值得优先做原作反事实核验。5局第四面失败的末次正常状态全部触发激光运动未知或地图中未被对象覆盖激光的弃权条件；这说明当前几何指导在该处证据不足，不等于已经证明神经网络漏看了全部激光。现版读入了激光线段、曲线边和已观测的直线端点运动；曲线节点运动仍明确标记未知。</p>
<h2>四、还存在训练流程因素，不能全怪奖励</h2>
<ul><li>同一组旧BC留出轨迹上，硬教师动作一致率约68.55%→37.52%；99个教师Bomb动作标签的最高概率匹配11/99→2/99。这里99个是Bomb动作标签，不能称为99个决死窗口：其中97个标签来自正常无保护状态。说明PPO后行为发生明显漂移，但不能据此判定决死能力下降。该留出集也不是新的独立实机测试；不同动作可能安全，一致率下降不能单独证明性能退化。</li>
<li>教师和训练奖励还存在策略偏好差异：30局教师数据共用508次Bomb，按当前严格口径只有25次可确认决死，483次属于普通或未确认；最终神经模型固定20局共用433次Bomb，其中408次可确认决死。必须核对“学习教师的完整策略”和“惩罚普通Bomb、优待决死”的目标是否一致，不能一面要求模仿教师，一面把教师大量使用的动作统一视为不良行为。</li>
<li>BC改变了4组GRU张量，但18个价值网络张量原样继承。第一批PPO的3个价值解释方差为负，第一轮更新KL约0.0645，存在价值网络未适应新隐状态的嫌疑。这是优先验证的流程问题，尚无因果对照。</li>
<li>“BC后、不加新PPO”的20局完整开发评估已经结束：1/20通关，平均推进5.1面，平均76,397原作帧、Miss 5.3、Bomb 16.7。它是神经网络，不是规则教师；使用另一组开发种子，不能直接与本页的独立测试20局排名或做因果判断。它证明单靠这次BC还不可靠，因此不能把全部问题归结为后续PPO遗忘。最终数据也已导出到TensorBoard独立开发组。</li>
<li>当前网络约506万参数，包含CNN、空间注意力、对象注意力和GRU256；它已经实机通关，容量不是首要证据。暂不靠继续加大网络掩盖奖励、覆盖和优化流程的问题。</li></ul>
<h2>五、新确认的风险分支校准缺口</h2>
<p>已逐张量核验：当前最终模型的点碰撞风险分支，仍与早期 action-pair-risk-v3 拟合结果完全一致，9个参数/缓冲张量未改变。其原始拟合选用8192个第一面状态、留出2048个第一面状态；所有人物圆判定分量均为2.0、矩形半尺寸均为1.0。位置和贴墙增广没有增加人物巨大化几何。第六面尾段出现的21.6圆判定和3.0矩形半尺寸，未在该拟合的原作来源状态中出现。有效合并半径可能与其他大子弹重叠，因此不能单凭这个事实断言所有大半径都没学过。</p>
<p>更直接的证据是：在第4、9、15、16局保存的真实前动作状态、原GRU记忆和最终权重上重新推理，动作概率与原记录的最大误差低于10⁻⁶。所选移动的风险网络预测余量分别约 +5.84、+3.19、+5.88、+5.65；当前几何模型分别为 −1.32、−0.34、−1.66、−2.64。已执行动作的下一份原作观测均在4帧后进入决死状态，确实发生了碰撞。其他候选方向没有实际执行，不能声称一定安全。问题已从“可能未读尺寸”缩小到“已读入尺寸，但冻结风险分支在巨大化状态下明显低估风险”，应优先校准，而非盲目放大惩罚系数。</p>
<p>3局漏决死还已验证：当时风险分支作用域不启用，Bomb logit未被风险分支改变，漏选出自基础策略本身。曲线激光的未知运动则不参与风险分支的方向性预测，只保留几何与未知标志供注意力和GRU使用；这也是下一轮的独立修复/验证点。</p>
<h2>六、下一轮的可执行顺序（提案，未实施）</h2>
<ol><li>冻结现有两模型，保留独立成绩；实机演示从已验证共同通关种子启动，明确不增加测试样本。</li>
<li>先补冻结风险分支的全尺寸、全关卡校准，重点校验巨大化人物和曲线激光；使用训练来源或有标记的合成几何，禁止把这13局独立测试失败片段直接并入训练后仍报原来的独立成绩。修正版与原版在相同开发种子上作完整实机对照。</li>
<li>BC-only完整开发已完成；后续用相同开发种子比较“PPO前价值校准/教师保持”与原流程，不能只靠不同种子的1/20和7/20判定PPO收益。已看的原测试集合保留历史口径，新一轮结论用新独立种子验收。</li>
<li>针对巨大化末段的真实有库存决死状态和第四面失败，补充同一单策略的关键状态训练覆盖；先验证神经网络输出，执行时仍不加入外部自动Bomb接管。</li>
<li>奖励另开有版本的对照：拉长信用分配、审查普通Bomb与决死的成本关系、以通关与净续命能力为主目标。先制定尺度、价值目标和评价口径，禁止把γ改掉却保留不匹配的势函数γ。任何改动需新训练对照，本轮不改。</li>
<li>主指标：自然开局、有限残机、禁止续关的六面通关率；辅助指标：每面消耗、获得的残机、可受伤暴露下的碰撞、可用决死漏选率、禁Bomb诊断。用开发集选择，最后再开新独立测试，并报告区间。</li>
<li>稳定策略确定后再移植果蝇：沿用同一原作接口和19动作，把单一强策略作为蒸馏教师；先验证感觉编码与脑内传播能表达躲避，再逐步放开允许训练的脑参数。不拆成多个关卡专家，不启动果蝇训练。</li></ol>
<h2>七、实机演示与后续使用</h2>
<p>原作真实窗口、实时CUDA推理的Lunatic六面演示已完成：6次Miss、26次Bomb，终局剩1残机、2个Bomb，模型权重未改变。使用预先验证的52158种子，不计入独立测试。第五面遇到原作暂停菜单后，使用仅对该进程生效的菜单输入恢复；没有接管战斗或自动救命。本次原作输入帧和推理次数包含暂停期间空转，不用于训练速度统计。</p>
<p>重新打开项目根目录“打开训练监控.cmd”，进入“模型实机演示”页；也可双击“打开模型实机演示.cmd”。支持选择模型、开始、暂停/继续、停止、打开记录与终局画面。旧模型和完整教师+PPO模型都保留，均7/20；默认旧模型仅因同批测试平均推进更远，不能称其通关率显著更好。新版演示接口会处理原作暂停菜单并把菜单输入单独计数，终局画面保留10分钟。</p>
<p class="small">生成时间：'''+html.escape(result['utc'])+'''。原始报告、模型SHA、13局数组SHA及离线几何诊断见同目录 stable-clear-wrapup-review-v1.json。本次奖励与模型优化均未修改。</p></main></html>'''
    page=r/'stable-clear-wrapup-review-v1.html';page.write_text(doc,encoding='utf-8')
    print(json.dumps(dict(json=str(target),html=str(page),cases=len(cases),
        failure_stages=result['observed_failure_stages'],reward_modified=False),ensure_ascii=False))


if __name__=='__main__':main()
