// QA_ONLY production script VM: no browser/video decode/inference claim.
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const root=__dirname+'/../../services/api/app/live/static/';
function page(){
  const elements=new Map();
  function element(id){
    const classes=new Set();
    return {id,value:'',textContent:'',children:[],src:'',disabled:false,
      classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x)},
      appendChild(x){this.children.push(x)},removeAttribute(){},pause(){},load(){}};
  }
  const document={cookie:'',getElementById(id){if(!elements.has(id)) elements.set(id,element(id));return elements.get(id)},
    createElement:element};
  const context={document,console,Date,JSON,encodeURIComponent,setInterval(){},clearInterval(){},XMLHttpRequest:class{}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(root+'common.js','utf8'),context);
  const html=fs.readFileSync(root+'upload.html','utf8');
  vm.runInContext([...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(x=>x[1]).join('\n'),context);
  vm.runInContext('roomId="QA_ONLY"',context);
  return {context,get:document.getElementById,eval:s=>vm.runInContext(s,context)};
}
(async()=>{
  const p=page();
  p.context.api=async()=>({inputs:[{id:'Camera1',label:'owner-provided recorded video'}]});
  await p.context.loadCatalog();
  assert.equal(p.get('catalogInput').children[0].value,'Camera1');
  assert.equal(p.get('selectInput').disabled,false);
  let call;
  p.get('catalogInput').value='Camera1';
  p.context.api=async(path,opts)=>{call={path,opts};return {sha256:'b'.repeat(64),meta:{w:320,h:240,fps:10,frame_count:15,duration_seconds:1.5}}};
  const oldGeneration=p.eval('pollGeneration');
  await p.get('selectInput').onclick();
  assert.equal(call.path,'/api/live/rooms/QA_ONLY/video/select');
  assert.equal(call.opts.body.id,'Camera1');
  assert.equal(p.get('sourceVideo').src,'/api/live/rooms/QA_ONLY/video/source?sha='+'b'.repeat(64));
  assert.ok(p.get('videoInfo').textContent.includes('15 frames'));
  assert.equal(p.eval('pollGeneration'),oldGeneration+1);
  assert.equal(p.get('feed').src,'');
  const unavailable=page(); unavailable.context.api=async()=>({inputs:[]});
  await unavailable.context.loadCatalog();
  assert.equal(unavailable.get('selectInput').disabled,true);
  assert.equal(unavailable.get('catalogInput').children[0].textContent,'Prepared inputs unavailable');
  const stale=page(); let resolve;
  stale.context.api=()=>new Promise(r=>resolve=r);
  const pending=stale.get('selectInput').onclick();
  stale.eval('roomId=null');
  resolve({sha256:'a'.repeat(64),meta:{}}); await pending;
  assert.equal(stale.get('sourceVideo').src,'');
  console.log('4 catalog UI contracts PASS: options, selection/source identity, unavailable, stale room ACK');
})().catch(e=>{console.error(e);process.exitCode=1});
