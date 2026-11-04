// Compiled in-process bulk read of pinned bullet/list fields. Binary records
// are (u32 address,u32 length,u8 bytes[length]). All reads are guarded by the
// OS, lists bounded, errors explicit. Other object types retain native reads.
const bulkCode = `
typedef unsigned int u32;
typedef unsigned char u8;
extern int __attribute__((stdcall)) rpm(void*,void*,void*,u32,u32*);
extern u8 output[2097152];
extern u32 state[2];
#define used state[0]
#define count state[1]
static int copy(u32 address,void* destination,u32 size) {
    u32 n=0;
    return address>=65536 && rpm((void*)-1,(void*)address,destination,size,&n) && n==size;
}
static int record(u32 address,u32 size) {
    if (used+8+size>sizeof(output)) return 0;
    if (!copy(address,output+used+8,size)) return 0;
    *(u32*)(output+used)=address; *(u32*)(output+used+4)=size;
    used+=8+size;count++;return 1;
}
int collect(void) {
    u32 manager,node,entry,next,iterations=0;
    used=0;count=0;
    if (!copy(0x4db530,&manager,4)) return -1;
    if (!manager) return 0;
    node=manager+124;
    while (node) {
        if (++iterations>8192) return -2;
        if (!record(node,8)) return -3;
        entry=*(u32*)(output+used-8); next=*(u32*)(output+used-4);
        if (entry) {
            if (!record(entry+0x20,8) || !record(entry+0x4d4,4)
                || !record(entry+0xbc0,0x50) || !record(entry+0x13bc,0x34)) return -4;
        }
        node=next;
    }
    return used;
}
void* data(void) {return output;}
int records(void) {return count;}
`;
if (Process.arch!=='ia32') throw new Error('Binary TH14 reader requires ia32');
const kernel=Process.getModuleByName('kernel32.dll');
const outputBacking=Memory.alloc(2097152),stateBacking=Memory.alloc(8);
const bulkModule=new CModule(bulkCode,{rpm:kernel.getExportByName('ReadProcessMemory'),
    output:outputBacking,state:stateBacking});
const collect=new NativeFunction(bulkModule.collect,'int',[],{scheduling:'cooperative'});
const data=new NativeFunction(bulkModule.data,'pointer',[]);
const records=new NativeFunction(bulkModule.records,'int',[]);
function listenBulk() {
    recv('binary-read',message=>{
        const request=message.payload;
        try {
            const length=collect();
            if (length<0) throw new Error('Compiled bullet capture failed: '+length);
            send({type:'binary-reply',request_id:request.request_id,pid:Process.id,
                checkpoint:request.checkpoint,records:records(),bytes:length},
                data().readByteArray(length));
        } catch(error) {
            send({type:'binary-error',request_id:request.request_id,detail:String(error.stack||error)});
        }
        listenBulk();
    });
}
listenBulk();
send({type:'binary-ready',pid:Process.id,compiled:true,format:'address-length-bytes-v1'});
