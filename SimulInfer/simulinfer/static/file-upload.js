// Match the existing file UI: quick submission, then short incremental polls.
export async function uploadFile(file,options,{url,onEvent,signal,error,httpError}){
  let task=null,finished=false,cursor=0,terminal=false;
  const form=new FormData();form.append('file',file);
  for(const [name,value] of Object.entries(options))form.append(name,value);
  try{
    const response=await fetch(url('translate/submit'),{method:'POST',body:form,signal});
    if(!response.ok)throw await httpError(response);
    task=(await response.json()).task_id;
    if(typeof task!=='string'||!task)throw error('uploadProtocol');
    const base=`translate/task/${encodeURIComponent(task)}`;
    while(!finished){
      const response=await fetch(url(`${base}?cursor=${cursor}`),{cache:'no-store',signal});
      if(!response.ok)throw await httpError(response);
      const state=await response.json();
      if(!Array.isArray(state.events)||!Number.isInteger(state.next_cursor)||
         !Number.isInteger(state.total_events)||state.next_cursor!==cursor+state.events.length||
         state.next_cursor>state.total_events||typeof state.done!=='boolean')throw error('uploadProtocol');
      for(const event of state.events){
        if(terminal)throw error('uploadProtocol');
        onEvent(event);
        if(event.event==='done')terminal=true;
      }
      cursor=state.next_cursor;
      finished=state.done&&cursor===state.total_events;
      if(finished&&!terminal)throw error('missingDone');
      if(!finished){
        await new Promise((resolve,reject)=>{
          if(signal?.aborted){reject(error('uploadCancelled'));return}
          const abort=()=>{clearTimeout(timer);reject(error('uploadCancelled'))};
          const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve()},cursor<state.total_events?0:500);
          signal?.addEventListener('abort',abort,{once:true});
        });
      }
    }
  }finally{
    if(task&&!finished){
      // Cancellation must reach the background job even when the UI fetch was aborted.
      fetch(url(`translate/task/${encodeURIComponent(task)}`),{method:'DELETE',keepalive:true}).catch(()=>{});
    }
  }
}
