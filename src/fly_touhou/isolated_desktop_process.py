"""Owned suspended Win32 child on a private, never-switched desktop.

This is process presentation isolation, not a simulator. Child desktop-binding
failures are fatal; input-desktop telemetry may be unavailable during a lock or
secure-desktop transition. There is no fallback to the user's input desktop.
Never calls SwitchDesktop,
SetThreadDesktop, desktop input, or foreground-restoration APIs.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import re
import uuid

CREATE_FLAGS=0x00000004|0x00000400|0x00000200|0x08000000
STARTUP_FLAGS=0x00000001|0x00000080  # USESHOWWINDOW + FORCEOFFFEEDBACK
DESKTOP_ACCESS=0x0001|0x0002|0x0004|0x0008|0x0040|0x0080  # no SWITCHDESKTOP


class STARTUPINFOW(ctypes.Structure):
    _fields_=[('cb',W.DWORD),('lpReserved',W.LPWSTR),('lpDesktop',W.LPWSTR),('lpTitle',W.LPWSTR),
        ('dwX',W.DWORD),('dwY',W.DWORD),('dwXSize',W.DWORD),('dwYSize',W.DWORD),
        ('dwXCountChars',W.DWORD),('dwYCountChars',W.DWORD),('dwFillAttribute',W.DWORD),
        ('dwFlags',W.DWORD),('wShowWindow',W.WORD),('cbReserved2',W.WORD),
        ('lpReserved2',ctypes.POINTER(ctypes.c_byte)),('hStdInput',W.HANDLE),
        ('hStdOutput',W.HANDLE),('hStdError',W.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_=[('hProcess',W.HANDLE),('hThread',W.HANDLE),('dwProcessId',W.DWORD),('dwThreadId',W.DWORD)]


class _WindowsAPI:
    def __init__(self):
        if os.name!='nt':raise OSError('Private native desktop requires Windows')
        self.user=ctypes.WinDLL('user32',use_last_error=True)
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        declarations=(
            (self.user,'GetProcessWindowStation',[],W.HANDLE),
            (self.user,'GetUserObjectInformationW',[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.POINTER(W.DWORD)],W.BOOL),
            (self.user,'OpenInputDesktop',[W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
            (self.user,'GetThreadDesktop',[W.DWORD],W.HANDLE),
            (self.user,'CreateDesktopW',[W.LPCWSTR,W.LPCWSTR,ctypes.c_void_p,W.DWORD,W.DWORD,ctypes.c_void_p],W.HANDLE),
            (self.user,'CloseDesktop',[W.HANDLE],W.BOOL),
            (self.kernel,'CreateProcessW',[W.LPCWSTR,W.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,W.BOOL,W.DWORD,
                ctypes.c_void_p,W.LPCWSTR,ctypes.POINTER(STARTUPINFOW),ctypes.POINTER(PROCESS_INFORMATION)],W.BOOL),
            (self.kernel,'ResumeThread',[W.HANDLE],W.DWORD),
            (self.kernel,'WaitForSingleObject',[W.HANDLE,W.DWORD],W.DWORD),
            (self.kernel,'TerminateProcess',[W.HANDLE,W.UINT],W.BOOL),
            (self.kernel,'CloseHandle',[W.HANDLE],W.BOOL),
        )
        for library,name,args,result in declarations:
            function=getattr(library,name);function.argtypes=args;function.restype=result

    def object_name(self,handle):
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        count=W.DWORD();buffer=ctypes.create_unicode_buffer(512)
        if not self.user.GetUserObjectInformationW(handle,2,buffer,ctypes.sizeof(buffer),ctypes.byref(count)):
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value

    def station_name(self):return self.object_name(self.user.GetProcessWindowStation())

    def input_desktop_name(self):
        handle=self.user.OpenInputDesktop(0,False,0x0001)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        try:return self.object_name(handle)
        finally:self.close_desktop(handle)

    def thread_desktop_name(self,thread_id):
        # GetThreadDesktop returns a borrowed handle: do not CloseDesktop it.
        return self.object_name(self.user.GetThreadDesktop(thread_id))

    def create_desktop(self,name):
        handle=self.user.CreateDesktopW(name,None,None,0,DESKTOP_ACCESS,None)
        if not handle:raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def create_suspended(self,executable,cwd,environment,desktop):
        info=STARTUPINFOW();info.cb=ctypes.sizeof(info);info.lpDesktop=desktop
        info.dwFlags=STARTUP_FLAGS;info.wShowWindow=0
        process=PROCESS_INFORMATION()
        command=ctypes.create_unicode_buffer('"'+str(executable)+'"')
        block=ctypes.create_unicode_buffer('\0'.join(key+'='+value for key,value in
            sorted(environment.items(),key=lambda row:row[0].upper()))+'\0\0')
        if not self.kernel.CreateProcessW(str(executable),command,None,None,False,CREATE_FLAGS,
                block,str(cwd),ctypes.byref(info),ctypes.byref(process)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'pid':int(process.dwProcessId),'thread_id':int(process.dwThreadId),
                'process_handle':process.hProcess,'thread_handle':process.hThread}

    def resume(self,handle):
        count=int(self.kernel.ResumeThread(handle))
        if count==0xffffffff:raise ctypes.WinError(ctypes.get_last_error())
        return count

    def wait(self,handle,milliseconds):return int(self.kernel.WaitForSingleObject(handle,milliseconds))

    def terminate(self,handle):
        if not self.kernel.TerminateProcess(handle,1):raise ctypes.WinError(ctypes.get_last_error())

    def close_handle(self,handle):
        if not self.kernel.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())

    def close_desktop(self,handle):
        if not self.user.CloseDesktop(handle):raise ctypes.WinError(ctypes.get_last_error())


class IsolatedDesktopProcess:
    def __init__(self,executable,*,cwd,env=None,api=None):
        self.api=api or _WindowsAPI();self.desktop_handle=None
        self.process_handle=self.thread_handle=None;self.pid=self.thread_id=None
        self.closed=False;self.resumed=False
        shared=os.environ.get('FLY_TOUHOU_DESKTOP_NAME')
        if shared is not None and not re.fullmatch(r'fly-touhou-job-[A-Za-z0-9_-]{1,96}',shared):
            raise ValueError('Invalid shared private desktop name; no default-desktop fallback')
        self.name=shared or ('fly-touhou-'+str(os.getpid())+'-'+uuid.uuid4().hex)
        self.audit={'schema':'th14-private-desktop-v1','desktop_name':self.name,
            'desktop_switched':False,'default_desktop_fallback':False,'resumed':False,
            'creation_flags':CREATE_FLAGS,'startup_flags':STARTUP_FLAGS,
            'handles_inherited':False,'input_desktop_separate':False,
            'shared_per_job':shared is not None}
        try:
            station=self.api.station_name()
            if station.casefold()!='winsta0':raise RuntimeError('Expected user interactive window station WinSta0')
            original_input=self._observe_input_desktop('creation')
            if original_input is not None and original_input.casefold()==self.name.casefold():raise RuntimeError('Private desktop is unexpectedly input desktop')
            self.desktop_handle=self.api.create_desktop(self.name)
            if self.api.object_name(self.desktop_handle)!=self.name:raise RuntimeError('Created desktop identity mismatch')
            environment=dict(os.environ)
            for key,value in (env or {}).items():
                for prior in tuple(environment):
                    if prior.casefold()==key.casefold():del environment[prior]
                environment[str(key)]=str(value)
            if any('\0' in key or '\0' in value for key,value in environment.items()):
                raise ValueError('Native environment contains NUL')
            desktop=station+'\\'+self.name
            child=self.api.create_suspended(Path(executable),Path(cwd),environment,desktop)
            for key,value in child.items():setattr(self,key,value)
            self.audit.update(pid=self.pid,thread_id=self.thread_id,desktop_path=desktop,
                input_desktop_at_creation=original_input,
                input_desktop_separate=True if original_input is not None else None)
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:error.add_note('Private child cleanup: '+repr(cleanup))
            raise

    def _observe_input_desktop(self, stage):
        try:
            return self.api.input_desktop_name()
        except OSError as error:
            if getattr(error, 'winerror', None) != 5:
                raise
            # A locked/secure desktop can deny this optional read. Do not
            # open it with more rights or switch to it. The child still has
            # to prove its own explicit private STARTUPINFO/thread binding.
            self.audit.setdefault('input_desktop_query_errors', {})[stage] = 5
            return None

    def resume_after_hooks(self,verified_startup_desktop):
        if self.closed or self.resumed:raise RuntimeError('Private child is closed or already resumed')
        current_input=self._observe_input_desktop('resume')
        self.audit.update(input_desktop_before_resume=current_input,
            remote_startup_desktop=verified_startup_desktop)
        if (not isinstance(verified_startup_desktop,str)
                or verified_startup_desktop.casefold()!=('WinSta0\\'+self.name).casefold()
                or (current_input is not None and current_input.casefold()==self.name.casefold())):
            raise RuntimeError('Private desktop startup binding failed before resume; no input-desktop fallback')
        count=self.api.resume(self.thread_handle)
        if count!=1:raise RuntimeError('Unexpected owned primary thread suspend count: '+str(count))
        self.resumed=True;self.audit['resumed']=True

    def verify_after_checkpoint(self,observed):
        # Before the primary thread enters user32 it has no desktop handle yet.
        # Its STARTUPINFO binding is checked before resume; now verify Win32's
        # actual thread assignment after the first original input checkpoint.
        if (not isinstance(observed,dict) or observed.get('schema')!='th14-main-thread-desktop-v1'
                or observed.get('verified') is not True
                or observed.get('queried_in_original_input_thread') is not True
                or observed.get('thread_id')!=self.thread_id):
            raise RuntimeError('Original input thread desktop verification is missing or failed')
        assigned=observed.get('name')
        # Original-thread assignment is retained in every checkpoint. The
        # unrelated foreground desktop is observed once, not polled on each
        # game read; changing/locking that desktop must not kill these games.
        current_input=(self.audit.get('input_desktop_after_checkpoint')
                       if self.audit.get('main_thread_assignment_verified') else
                       self._observe_input_desktop('first_checkpoint'))
        self.audit.update(main_thread_desktop=assigned,input_desktop_after_checkpoint=current_input)
        if (not isinstance(assigned,str) or assigned.casefold()!=self.name.casefold()
                or (current_input is not None and current_input.casefold()==self.name.casefold())):
            raise RuntimeError('Private desktop isolation failed at native checkpoint')
        self.audit['native_main_thread_verification']=dict(observed)
        self.audit['main_thread_assignment_verified']=True

    def close(self):
        if self.closed:return
        errors=[]
        try:
            if self.process_handle:
                wait=self.api.wait(self.process_handle,0)
                if wait==0x102:
                    try:self.api.terminate(self.process_handle)
                    except OSError:
                        # The existing Job/device close can complete concurrently.
                        # Access denied is harmless only if this exact handle has
                        # now become signaled; never assume a PID was terminated.
                        if self.api.wait(self.process_handle,2000)!=0:raise
                    wait=self.api.wait(self.process_handle,2000)
                self.audit['exact_owned_exit_confirmed']=wait==0
                if wait!=0:errors.append(RuntimeError('Private child exit not confirmed: '+str(wait)))
        except BaseException as error:errors.append(error)
        finally:
            for field in ('thread_handle','process_handle'):
                handle=getattr(self,field)
                if handle:
                    try:self.api.close_handle(handle)
                    except BaseException as error:errors.append(error)
                    setattr(self,field,None)
            if self.desktop_handle:
                try:self.api.close_desktop(self.desktop_handle)
                except BaseException as error:errors.append(error)
                self.desktop_handle=None
            self.closed=True;self.audit['closed']=True
        if errors:raise RuntimeError('Private desktop cleanup failed: '+repr(errors))
