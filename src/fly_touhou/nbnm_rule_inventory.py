"""Human-readable exhaustive rule inventory; no claim of a complete emulator.

The original bytecode stays in the external catalogue. Each entry below has
source sites and raw parameters, including variables rather than guessed values.
"""
from collections import Counter
from pathlib import Path
import csv
import json
import struct

from .nbnm_rule_catalog import EX_OPS, literal

# Names describe native effects, not merely the on-screen appearance.
EX_ANALYSIS = {
    1: ('初始速度变化', '激活内部加速状态；不能把刚出生的速度永久外推。', '定时器、初始速度、当前激活标志'),
    2: ('出生动画／释放', '出生位置回退四倍速度，释放前运动与判定状态不同；出现不等于立刻可碰撞。', '动画门、原生状态、出生计时'),
    4: ('向量加速度', '改变平面速度；方向参数可引用当前角度或瞄准方向。', '速度、加速度、持续时间、角度哨兵值'),
    8: ('极坐标加速度', '同时改变速度和角度，产生弯曲轨迹。', '每帧速度增量、角增量、持续时间'),
    16: ('分段转向／再次瞄准', '停顿或运动一段时间后换向，可能重新瞄准自机。', '转向间隔、次数、模式、转向时的自机位置'),
    64: ('边界反弹', '出界后反射方向；不能在第一次离开屏幕时直接删除。', '反弹次数、边界掩码、实际场地边界'),
    128: ('暂时关闭碰撞', '弹幕仍存在和运动，但可碰撞状态有独立时间。', '保护计时、原生状态与标志'),
    256: ('屏外保留', '延迟屏外回收；回到场内的弹幕仍是威胁。', '保留时间、回收条件'),
    512: ('换弹型／颜色／判定', '不能只换贴图；原生类型表的碰撞几何也可能变化。', '新类型、颜色、切换帧、原生碰撞尺寸'),
    1024: ('删除／消弹', '生命周期终止；预测不能继续留下幽灵障碍。', '执行顺序、删除状态'),
    2048: ('播放声音', '本指令本身不改变碰撞；执行后仍须读取后面的运动指令。', '后续程序位置'),
    4096: ('屏幕穿越', '越过一侧后从另一侧返回；坐标跳变不代表对象变快。', '次数、边界掩码、穿越后的坐标'),
    8192: ('子弹再发弹', '以当前子弹为发射器生成子弹；读取下一条配套记录，可能删除母弹。', '母弹位置／角度／速度、发射模式、下一条记录、随机数'),
    16384: ('再发弹配套记录', '与前一条再发弹指令一起解释，提供弹型等数据；不能算成第二次独立发射。', '与前一条记录配对、子弹类型、母弹删除参数'),
    32768: ('原生反应函数选择', '写入子弹反应编号；六面距离反应不能由匀速公式代替。', '反应编号、自机位置、原生回调实现'),
    65536: ('EX 程序跳转', '循环和分支改变下一段运动，不能把 EX 当成一次性平铺参数。', '目标索引、活跃状态、执行条件'),
    1048576: ('动画显示标志', '本指令修改动画标志；不等于新增碰撞或免疫。', '后续运动程序、原生碰撞标志'),
    2097152: ('速度／方向插值', '在指定时间内改变速度及方向；角度可以使用保留／瞄准的特殊参数。', '起终速度、方向、持续时间、自机位置'),
    4194304: ('碰撞尺寸插值', '六面大球的判定随插值放大／缩小；两浮点数是插值两端，不可只读当前半径。', '基础几何、起终缩放、插值模式、时间、自机判定尺寸'),
    33554432: ('绘制层', '影响显示层次；不能据此改变危险程度。', '后续程序'),
    67108864: ('HOLD／延迟释放', '与 WAIT 分开维护状态及计时，延迟结束后可能立即接新运动程序。', '保持标志、计时、后续 EX'),
    2147483648: ('等待／程序同步', '暂停后续 EX 的执行；并不普遍意味着子弹速度变零。', '等待计时、并行条件、当前速度'),
}
AIM_ANALYSIS = {
    0: ('自机狙扇形', '使用发射那一刻的自机位置，候选移动路线会改变瞄准角。'),
    1: ('固定角扇形', '角度、扇宽、弹数和层间速度由规则设置。'),
    2: ('自机狙圆环', '圆环整体转向自机；仍须预测整环。'),
    3: ('固定圆环', '规则控制旋转相位与层间转角。'),
    4: ('自机狙错位圆环', '圆环有半间隔偏移，不能与模式 2 合并。'),
    5: ('固定错位圆环', '在固定圆环基础上偏移半个弹间隔。'),
    6: ('随机角度', '原作支持；本次六面 Lunatic 静态脚本未直接设置。'),
    7: ('随机速度圆环', '原作支持；本次六面 Lunatic 静态脚本未直接设置。'),
    8: ('随机角度与速度', '依赖共享 RNG 的后续调用顺序；读到当前种子不等于全局未来已复现。'),
    9: ('自机狙多层交错', '原作支持；本次六面 Lunatic 静态脚本未直接设置。'),
    10: ('固定多层交错', '层间交错角度与速度，原生顺序影响随机数和生成次序。'),
    11: ('双叶速度分布', '原作支持；本次六面 Lunatic 静态脚本未直接设置。'),
    12: ('错位双叶速度分布', '同一发射器内不同方向的速度不同，不能把所有子弹设为统一速度。'),
}
OP_NAMES = {
    600:'重置发射器',601:'生成普通弹',602:'设置弹型与颜色',603:'发射原点偏移',604:'角度与角间隔',
    605:'两端速度',606:'弹数与层数',607:'发射模式',608:'发射声音',609:'写短 EX 记录',610:'写长 EX 记录',
    611:'追加短 EX 记录',612:'追加长 EX 记录',613:'清除弹幕与激光',614:'复制发射器',
    615:'范围消弹（原生模式1）',616:'范围消弹（原生模式0）',624:'按难度选择速度',625:'按难度选择弹数',
    626:'极坐标发射偏移',627:'环形出生偏移',628:'绝对发射位置',629:'Boss 外圈显示参数',
    630:'Boss 显示类型',632:'安装／移除原生逐帧回调',633:'专用状态开关',637:'调用专用原生效果',
    641:'退回一格 EX 写入位置',700:'激光长度／宽度参数',701:'激光时间／节点参数',
    702:'生成直线激光',703:'生成另一类直束',705:'激光相关生成',711:'生成曲线激光',
}
# Field types for every projectile opcode present in the extracted scripts.
SIGNATURES = {600:'i',601:'i',602:'iii',603:'iff',604:'iff',605:'iff',606:'iii',607:'ii',608:'iii',
    609:'iiiiiiff',610:'iiiiiiiiffff',611:'iiiiiff',612:'iiiiiiiffff',613:'',614:'ii',615:'f',616:'f',
    624:'iffffffff',625:'iiiiiiiii',626:'iff',627:'if',628:'iff',629:'fi',630:'i',632:'i',633:'i',637:'i',
    641:'i',700:'iffff',701:'iiiiii',702:'i',703:'i',705:'i',711:'i'}


def argument_description(row):
    signature=SIGNATURES.get(row['opcode'])
    if signature is None:return 'raw:'+row['raw_hex'][32:]
    raw=bytes.fromhex(row['raw_hex']);answer=[]
    for index,kind in enumerate(signature):
        value=struct.unpack_from('<'+kind,raw,16+index*4)[0]
        text=format(value,'.8g') if kind=='f' else str(value)
        answer.append(('变量/栈引用:' if row['parameter_mask']&(1<<index) else '')+text)
    return ', '.join(answer)


def build_inventory(catalog):
    sites=[];style_sources=[]
    for program in catalog['programs']:
        for sub in program['subroutines']:
            for ins in sub['instructions']:
                if not ins['difficulty_mask']&8 or not 600<=ins['opcode']<800:continue
                site=dict(file=program['file'],subroutine=sub['name'],offset=ins['offset'],
                    local_time=ins['time'],opcode=ins['opcode'],name=OP_NAMES.get(ins['opcode'],'未命名'),
                    parameter_mask=ins['parameter_mask'],arguments=argument_description(ins))
                if ins['opcode'] in EX_OPS:
                    ci=3 if ins['opcode'] in (609,610) else 2
                    value=literal(ins,ci)
                    site['ex_command']=None if value is None else value&0xffffffff
                    if site['ex_command'] in (512,16384):
                        style_sources.append(dict(file=site['file'],subroutine=site['subroutine'],
                            offset=site['offset'],style=literal(ins,ci+1),source='EX-'+hex(site['ex_command'])))
                if ins['opcode']==602:
                    style_sources.append(dict(file=site['file'],subroutine=site['subroutine'],
                        offset=site['offset'],style=literal(ins,1),source='shooter'))
                sites.append(site)
    return dict(schema='th14-six-stage-hazard-inventory-v1',sites=sites,style_sources=style_sources,
        all_literal_styles=sorted({s['style'] for s in style_sources if s['style'] is not None}),
        static_programs_complete=True,forward_simulation_complete=False,
        notes=['Includes debug/helper functions; counts are not distinct spell cards.',
               'All rank masks retained in catalogue; this index selects Lunatic.',
               'Call-local time is not an absolute stage frame.'])


def write_analysis(catalog,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inventory=build_inventory(catalog);sites=inventory['sites'];summary=catalog['summary']
    (output/'rule_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'所有弹幕指令与参数.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','subroutine','offset','local_time','opcode','name','parameter_mask','arguments','ex_command'])
        writer.writeheader();writer.writerows(sites)
    lines=['# 辉针城六面 Lunatic 全部弹幕规则分析','',
        '范围：原作六面道中、中 Boss、Boss、第四面两个角色分支及 default 公共脚本；不是从教师走过的轨迹抽样。',
        f"提取 {len(catalog['programs'])} 个脚本、{summary['function_count']} 个子程序、{summary['instruction_count']} 条 Lunatic 指令。包含辅助和调试子程序，不能解释为同样数量的符卡。",
        f"直接发射调用点 {summary['emission_sites']} 个；弹幕／激光配置及控制指令 {len(sites)} 条。静态调用点不等于实机弹数。",
        '完整原始参数见 [所有弹幕指令与参数.csv](所有弹幕指令与参数.csv)，逐子程序索引见 [全部弹幕规则清单.md](全部弹幕规则清单.md)。',
        '', '## 一、按六面分解', '', '| 关卡 | 普通弹发射点 | 直线激光发射点 | 曲线激光发射点 | 运动/判定程序类型 |','|---|---:|---:|---:|---|']
    for stage in range(1,7):
        rows=[r for r in sites if r['file'].startswith(f'st0{stage}')]
        count=Counter(r['opcode'] for r in rows)
        ex=sorted({r['ex_command'] for r in rows if r.get('ex_command') is not None})
        lines.append(f"| {stage} | {count[601]} | {count[702]} | {count[711]} | "+'、'.join(EX_ANALYSIS[e][0] for e in ex)+' |')
    lines += ['', '第一面没有直接激光生成指令；第三面道中存在一处直线激光生成指令。第四面两条分支不能合并成同一个 Boss 脚本。',
        '上表统计不含 default 公共脚本中的发射点；所有未执行分支仍保留。', '', '## 二、发射排列与瞄准', '',
        '| 模式 | 含义 | 预测方式 |','|---|---|---|']
    for mode in summary['aim_modes']:
        name,analysis=AIM_ANALYSIS[mode];lines.append(f'| {mode} | {name} | {analysis} |')
    lines += ['', '六面脚本直接使用 9 种模式；原作发射器还支持另外 4 种，不能把“本次没有直接设置”写成引擎不支持。',
        '', '## 三、全部 22 类运动、尺寸、判定及生命周期指令','',
        '| EX 编号 | 作用 | 对路径规划的影响 | 需要维护的状态 | 出现关卡 |','|---|---|---|---|---|']
    for command in summary['ex_command_types']:
        name,why,state=EX_ANALYSIS[command]
        locations=sorted({r['file'][3] for r in sites if r.get('ex_command')==command and r['file'].startswith('st0')})
        lines.append(f"| 0x{command:X} | {name} | {why} | {state} | {','.join(locations)} |")
    lines += ['', 'EX 按顺序和活跃标志执行，condition 允许部分指令并发。WAIT、HOLD、缩放、转向及程序跳转必须组成状态机。',
        '0x2000 与后续 0x4000 配套记录一起构成再发弹；后者不能误计为一次单独发射。',
        '', '## 四、激光必须单独建模','',
        '1. 直线激光：维护原点、朝向、当前长度、最大长度、移动速度、行程上限、宽度、有效状态。先增长到长度上限再平移，与普通弹不同。原生碰撞还裁去两端，并使用自身的宽度换算。',
        '2. 曲线激光：维护轨迹记录及历史节点，每个节点对应不同出生时刻；只用头尾或当前速度会漏掉弯曲部分。四面出现 37 个静态生成点。',
        '3. 激光专用 EX：与普通弹共用编号不代表更新语义相同；必须按激光类的原生执行函数解释，不能复用普通弹状态机冒充完整支持。',
        '4. 700/701 只是配置；702/711 才是这批脚本实际使用的生成指令。703/705 虽属于引擎范围，本次目录没有直接调用。',
        '', '## 五、大球、变形与碰撞几何','',
        '大球是“原生弹型几何 × 当前尺寸插值 × 判定状态”，不是“看起来很大的圆”。EX 0x400000 的两端缩放、时间及插值模式必须保留；换弹型后还要更新基础判定。ANM 动画释放及动态几何也需与原生碰撞代码一起核对，仅提取 ECL 尚不足以覆盖这些行为。',
        '六面自机也可能放大，未来判定必须同时考虑自机尺寸。五面画面翻转不能直接当成物理坐标翻转，需要保留逻辑坐标与输入变换。',
        f"直接设置及换型/再发弹配套记录中的常量弹型并集：{inventory['all_literal_styles']}。变量弹型保留引用，不猜值。",
        '', '## 六、弹幕外还必须预测的规则','',
        '- 新敌人生成、移动发射器、隐藏的 StageMain、异步子程序、局部计时、难度选择。',
        '- 敌人被击破后的死亡攻击、Boss 血量/时间中断、切换卡时消弹。玩家的射击会改变这些事件时刻。',
        '- 整屏/范围消弹、类型变化、屏外回收、实际弹池/激光池容量。',
        '- 敌人本体碰撞、不可碰撞到可碰撞的切换、保护时段以及自机判定半径。',
        '- 原生回调：四面反应对象；五面翻转；六面尺寸及距离相关反应。不能仅有 ECL 指令表而忽略 EXE 中的函数。',
        '', '| 原生回调/效果调用 | 参数 | 脚本位置 |','|---|---|---|']
    for row in sites:
        if row['opcode'] in (632,633,637):
            lines.append(f"| {row['opcode']} | {row['arguments']} | {row['file']}:{row['subroutine']} +0x{row['offset']:X} |")
    lines += ['', '## 七、当前代码覆盖与实施次序','',
        '| 层次 | 当前状态 | 下一步判据 |','|---|---|---|',
        '| 全脚本规则目录 | 已提取并保留全部字节码、变量引用、难度分支和参数 | 新增规则不能只靠看过的轨迹补丁 |',
        '| 未来生成事件 | 原生发射器、部分 ECL、候选位置瞄准、RNG 算法和新敌人预演已实现 | 发射时刻/原点/初速度与原作逐帧对照 |',
        '| 已出现弹幕 | 沿用已有原生读取和运动模块；不是本轮新增的全规则引擎 | 分别记录有效预测长度和未覆盖的后续 EX |',
        '| 尚未生成弹幕的完整运动 | 尚未完整；出生动画、全部 EX 组合、激光生命周期、回调仍有未覆盖项 | 按上表所有规则族和组合补齐原作对照 |',
        '| 完整未来 RNG 和阶段切换 | 尚未完整；不能承诺随机弹最终落点已全部预知 | 全局执行顺序、击破/死亡分支和容量共同复现 |',
        '| 接入网络 | 独立未来特征通道，新增接口零初始化；默认关闭 | 不改变脑拓扑，迁移保留已有参数；先验证输入再试训 |',
        '', '实施顺序：发射/瞄准与出生 → 普通弹 EX 状态机 → 直线和曲线激光 → 六面缩放/距离回调与自机尺寸 → 死亡/阶段分支及全局 RNG → 与原作比对后做冻结对照。',
        '不同规则族可复用已有模块，但不以首面几个预测正确代替全部六面的覆盖。',
        '', '## 八、供模型使用的危险信息','',
        '输入以当前原生状态、ECL 局部时钟/程序位置、对象状态和候选玩家轨迹为条件。输出分为已观测危险、规则预告危险、预测有效长度与未覆盖标志。',
        '精确复现、保守包络、条件预测、未知四种状态分开记录。未知不能被编码成已确认安全，保守包络也不能当成真实碰撞标签。',
        '继续用单个连接组网络执行动作；规则模块提供感觉和训练指导，不把教师实际接管的数据混入学生 PPO。',
        '', '## 依据','',
        '本机 TH14 1.00b 原始 ECL；发射器 4191A0/417660；EX 执行 418140；缩放 418E61/41678D；直线激光 43ACA0/43BE20/43C250；曲线轨迹 440F60/441430；RNG 403310。',
        '外部工具：[thtk](https://github.com/thpatch/thtk)；编号命名参考 [MERLIN](https://github.com/Priw8/merlin/blob/master/merlin/th16/globals/etama.tecl)。MERLIN 面向其他作品，TH14 的实际语义以本地代码为准。',
    ]
    (output/'全部弹幕规则分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(projectile_instruction_sites=len(sites),literal_styles=inventory['all_literal_styles'],
        ex_families=len(summary['ex_command_types']),forward_simulation_complete=False)
