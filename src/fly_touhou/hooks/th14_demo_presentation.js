// Only appended for real-rendered, owned visible demonstrations. Training's
// private-desktop hook and every gameplay action remain unchanged.
const demoPresentationAudit = {schema:'th14-visible-demo-no-input-v2',
    keyboard_queries_blocked:0,focus_loss_messages_normalized:0,
    physical_input_generated:false,gameplay_action_overrides:0};
for (const name of ['GetAsyncKeyState','GetKeyState']) {
    replaceExport('user32.dll',name,() => {
        demoPresentationAudit.keyboard_queries_blocked++;return 0;
    },'int',['int']);
}
Interceptor.attach(at(hook.window_proc.rva), {
    onEnter(args) {
        const message=args[1].toUInt32(),value=args[2].toUInt32();
        // Normalize only the owned game's view of activity; Windows keeps the
        // real foreground window. Never call SetForegroundWindow or SetFocus.
        if (message===0x0006 && (value&0xffff)===0) {
            args[2]=ptr(1);demoPresentationAudit.focus_loss_messages_normalized++;
        }
        if (message===0x0008) {
            args[1]=ptr(0);demoPresentationAudit.focus_loss_messages_normalized++;
        }
    }
});
rpc.exports.demopresentationaudit = () => ({...demoPresentationAudit});
rpc.exports.demoinputguardprobe = () => {
    // Invoke only these owned-process replacements; never generate a key.
    const values=[];
    for (const name of ['GetAsyncKeyState','GetKeyState']) {
        const fn=new NativeFunction(exported('user32.dll',name),'int',['int'],
            {abi:'stdcall',scheduling:'cooperative'});
        values.push(fn(0x1b));
    }
    return {values,physical_input_generated:false};
};
