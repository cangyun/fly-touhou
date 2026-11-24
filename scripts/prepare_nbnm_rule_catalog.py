"""Extract and catalogue every original stage 1..6 ECL rule on local storage."""
from pathlib import Path
import argparse,collections,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--thdat',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from fly_touhou.nbnm_rule_catalog import write_catalog
    from fly_touhou.native_shot_rules import EX_NAMES,AIM_NAMES
    a.output.mkdir(parents=True,exist_ok=True);rules=a.output/'rules';rules.mkdir(exist_ok=True)
    listing=subprocess.run([str(a.thdat),'-l','14',str(a.archive)],capture_output=True,check=True).stdout.decode('cp932')
    names=re.findall(r'^(?:default|st0[1-6][a-z0-9]*)\.ecl(?=\s)',listing,re.M)
    subprocess.run([str(a.thdat),'-x','14',str(a.archive),*names],cwd=rules,check=True,capture_output=True)
    result=write_catalog(rules,a.output/'catalog.json');s=result['summary']
    from fly_touhou.nbnm_rule_inventory import write_analysis
    write_analysis(result,a.output)
    rows=s['functions'];lines=['# TH14 六面完整规则清单',
        '',f"原作脚本 {len(result['programs'])} 个；Lunatic 子程序 {s['function_count']} 个；指令 {s['instruction_count']} 条。",
        '涵盖六面道中、中 Boss、Boss，以及第四面两个分支。清单来自脚本字节码，不来自曾经执行的教师路线。',
        '静态调用点数不等于实机发射次数；子程序含辅助和调试程序，不代表同样数量的符卡。循环、分支、死亡攻击和难度掩码保留在 catalogue 中。',
        '', '## 发射模式', '', '| 编号 | 模式 | 使用位置 |','|---|---|---|']
    for mode in s['aim_modes']:
        sites=[r['file']+':'+r['subroutine'] for r in rows if mode in r['aim_modes']]
        lines.append(f"| {mode} | {AIM_NAMES.get(mode,'待核对')} | {'、'.join(sites)} |")
    lines+=['','## 弹幕运动、判定与再发射程序','','| EX 编号 | 含义 | 静态设置数 | 使用文件 |','|---|---|---|---|']
    for command in s['ex_command_types']:
        sites=[r for r in rows if any(e['command']==command for e in r['ex_commands'])]
        count=sum(sum(e['command']==command for e in r['ex_commands']) for r in sites)
        lines.append(f"| 0x{command:X} | {EX_NAMES.get(command,'待核对')} | {count} | {', '.join(sorted({r['file'] for r in sites}))} |")
    lines+=['','## 激光','','700、701 是参数配置，702 是直线激光生成，711 是曲线激光生成。必须区分生成/长度变化/宽度/预告与有效判定，曲线保留历史节点，不能仅用头尾连线代替。',
        '', '| 文件与子程序 | 激光指令 |','|---|---|']
    for r in rows:
        laser={int(k):v for k,v in r['opcodes'].items() if 700<=int(k)<800}
        if laser:lines.append(f"| {r['file']}:{r['subroutine']} | {laser} |")
    lines+=['','## 大球与尺寸','','球形、米弹、音符、激光等外观编号不能替代碰撞定义。应读取原生类型表的碰撞尺寸；EX 0x400000 改变尺寸，0x200 改变类型时可能同时改变判定。六面还需保留人物判定半径与缩放，距离反应回调，以及敌人自身接触判定。',
        '',f"Lunatic 脚本直接使用的弹型编号：{s['bullet_styles']}。",'',
        '## 全部子程序索引','','| 文件 | 子程序 | 发弹点 | EX 种类 | 依赖变量 | 调用/生成目标 |','|---|---|---|---|---|---|']
    for r in rows:
        ex=','.join(f"0x{x:X}" for x in sorted({e['command'] for e in r['ex_commands'] if e['command'] is not None}))
        calls=','.join(sorted({c['target'] for c in r['calls']}))
        lines.append(f"| {r['file']} | {r['subroutine']} | {len(r['emit_sites'])} | {ex} | {r['variable_dependencies']} | {calls} |")
    lines+=['','## 实现覆盖边界','',
        '已完成：全量原作规则预加载；完整字节码、局部时间、难度掩码、变量引用与调用边保留；原生发射器读取；按候选玩家轨迹执行已支持的 ECL，产生未来发弹事件。',
        '不能宣称全部前向模拟已完成：新敌人调度、死亡/血量中断、共享随机数的全局消耗顺序、完整 EX 状态机、激光生成后的生命周期、特殊原生回调仍需逐项完成原作对照。',
        '预测器必须报告这些边界；未处理指令不得被算作“没有危险”。静态规则目录完整，与物理模拟完整是两项不同的指标。',
        '', '命名参考：MERLIN 的 etama 常量；TH14 实际出现的编号以本地原作字节码为准。',
        'https://github.com/Priw8/merlin/blob/master/merlin/th16/globals/etama.tecl']
    (a.output/'全部弹幕规则清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in s.items() if k not in ('functions','opcodes')},ensure_ascii=False))

if __name__=='__main__':main()
