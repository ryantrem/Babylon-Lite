/** Exact ordered paragraph wrapping with bounded replay scratch. */
export const COMPUTE_TEXT_CHUNK_LAYOUT = `
@group(0) @binding(0) var<storage,read> params: array<u32>;
@group(0) @binding(1) var<storage,read> shaped: array<i32>;
@group(0) @binding(2) var<storage,read> flags: array<u32>;
@group(0) @binding(3) var<storage,read> slots: array<u32>;
@group(0) @binding(4) var<storage,read_write> instances: array<u32>;
@group(0) @binding(5) var<storage,read_write> info: array<u32>;
@group(0) @binding(6) var<storage,read_write> carry: array<u32>;
@group(0) @binding(7) var<storage,read_write> scratch: array<u32>;
const NONE:u32 = 0xffffffffu;
// Wrap state words 0..13; carry[16..29] and info quadrants 0/32/64/96 persist them.
const STATE:u32 = 14u;
var<private> state:array<u32,STATE>;
var<private> events:u32;
var<private> replay:bool;
fn flag(c:u32)->u32 {
  let at=c+1u;
  return (flags[at/4u] >> ((at%4u)*8u)) & 255u;
}
fn fail(code:u32) {
  if (carry[0] == 0u || params[11] < carry[1]) { carry[0]=code; carry[1]=params[11]; }
}
fn event(end:u32) {
  if (replay) {
    let at=4u+params[0]*4u+events*8u;
    scratch[at]=state[2]; scratch[at+1u]=end; scratch[at+2u]=state[1];
    scratch[at+3u]=state[4];
    scratch[at+4u]=bitcast<u32>(max(0.0,f32(bitcast<i32>(state[7])-bitcast<i32>(state[4]))*bitcast<f32>(params[9])));
    scratch[at+5u]=state[3]; scratch[at+6u]=0u; scratch[at+7u]=0u;
    events++;
  }
  state[1]++;
}
fn scan() {
  let count=info[128];
  var visible=0u;
  for (var i=0u;i<count;i++) {
    let at=4u+i*6u;
    let ordinal=state[0];
    let cluster=params[1]+u32(shaped[at+1u]);
    let f=flag(u32(shaped[at+1u]));
    let advance=shaped[at+2u];
    let pen=bitcast<i32>(state[5]);
    if (replay) {
      let p=4u+i*4u;
      scratch[p]=state[5]; scratch[p+1u]=state[6]; scratch[p+2u]=visible; scratch[p+3u]=0u;
    }
    if ((f&1u)==0u) {
      let end=pen+advance;
      loop {
        if (ordinal==state[2] || f32(end-bitcast<i32>(state[4]))*bitcast<f32>(params[9])<=bitcast<f32>(params[6])) { break; }
        if (state[8]!=NONE) {
          let content=state[7]; state[7]=state[10];
          event(state[8]+1u);
          state[7]=content; state[2]=state[8]+1u; state[3]=state[13];
          state[4]=state[9]; state[8]=NONE;
          continue;
        }
        if (cluster!=state[11] && (f&4u)==0u) {
          event(ordinal); state[2]=ordinal; state[3]=cluster; state[4]=state[5];
        }
        break;
      }
      state[7]=bitcast<u32>(end);
    }
    state[5]=bitcast<u32>(pen+advance);
    state[6]=bitcast<u32>(bitcast<i32>(state[6])+shaped[at+3u]);
    var next=info[130];
    var before=info[131];
    if (i+1u<count) {
      next=params[1]+u32(shaped[at+7u]);
      before=flag(u32(shaped[at+7u])-1u);
    }
    if (next!=NONE && next>cluster && (before&2u)!=0u) {
      state[8]=ordinal; state[9]=state[5]; state[10]=state[7]; state[13]=next;
    }
    state[11]=cluster; state[0]++;
    if (slots[u32(shaped[at])]!=0xfffffffeu) { visible++; state[12]++; }
  }
  if (params[5]!=0u) { event(state[0]); }
}
@compute @workgroup_size(1)
fn summary() {
  if (params[5]!=0u) { carry[6]=NONE; carry[7]=0u; }
  info[130]=carry[6]; info[131]=carry[7];
  var status=u32(shaped[0]);
  let count=u32(shaped[1]);
  if (status==0u && count>params[0]) { status=2u; }
  var visible=0u;
  if (status==0u) {
    for (var i=0u;i<count;i++) {
      let glyph=u32(shaped[4u+i*6u]);
      if (glyph>=params[10]) { status=3u; break; }
      let slot=slots[glyph];
      if (slot==NONE) { status=3u; break; }
      if (slot!=0xfffffffeu) { visible++; }
    }
  }
  info[128]=select(0u,count,status==0u); info[129]=visible; info[132]=status;
  if (status!=0u) { fail(status); }
  if (count!=0u && status==0u) {
    let c=u32(shaped[5]);
    carry[6]=params[1]+c;
    carry[7]=flag(c-1u);
  }
}
@compute @workgroup_size(1)
fn measure() {
  if (params[4]!=0u) {
    for (var i=0u;i<STATE;i++) { carry[16u+i]=0u; }
    carry[24]=NONE; carry[27]=NONE;
    carry[19]=select(select(0u,info[130],info[130]!=NONE),params[1]+u32(shaped[5]),info[128]!=0u);
  }
  info[133]=carry[4];
  var equal=params[12]==0u && info[134]==params[6] && info[135]==params[9] && info[136]==info[130] && info[137]==info[131];
  for (var i=0u;i<STATE;i++) { state[i]=carry[16u+i]; info[i]=state[i]; equal=equal && info[64u+i]==state[i]; }
  if (equal && info[138]!=0u) {
    for (var i=0u;i<STATE;i++) { state[i]=info[96u+i]; }
    info[139]=0u;
  } else {
    replay=false; scan(); info[139]=1u;
  }
  for (var i=0u;i<STATE;i++) { info[32u+i]=state[i]; carry[16u+i]=state[i]; info[64u+i]=info[i]; info[96u+i]=state[i]; }
  info[134]=params[6]; info[135]=params[9]; info[136]=info[130]; info[137]=info[131]; info[138]=1u;
  if (params[5]!=0u) { carry[4]+=state[1]; }
}
@compute @workgroup_size(1)
fn replayChunk() {
  if (params[5]!=0u) { carry[48]=0u; }
  for (var i=0u;i<STATE;i++) { state[i]=info[i]; }
  let first=state[0];
  replay=true; events=0u; scan();
  for (var i=0u;i<STATE;i++) { if (state[i]!=info[32u+i]) { fail(2u); } }
  let incoming=carry[48];
  var same=params[12]==0u && info[139]==0u;
  for (var i=0u;i<17u;i++) {
    let value=select(0u,carry[48u+i],i<=incoming*8u);
    same=same && info[160u+i]==value;
    info[160u+i]=value;
  }
  for (var e=0u;e<incoming;e++) {
    let at=4u+params[0]*4u+events*8u;
    for (var j=0u;j<8u;j++) { scratch[at+j]=carry[49u+e*8u+j]; }
    events++;
  }
  scratch[0]=events;
  var outgoing=0u;
  for (var e=0u;e<events;e++) {
    let at=4u+params[0]*4u+e*8u;
    if (scratch[at]<first) {
      if (outgoing>=2u) { fail(2u); break; }
      for (var j=0u;j<8u;j++) { carry[49u+outgoing*8u+j]=scratch[at+j]; }
      outgoing++;
    }
  }
  carry[48]=outgoing;
  for (var i=0u;i<7u;i++) {
    var value=params[6u+i];
    if (i==4u) { value=info[133]; }
    if (i==5u) { value=params[3]; }
    if (i==6u) { value=params[13]; }
    same=same && info[180u+i]==value; info[180u+i]=value;
  }
  info[140]=select(1u,0u,same);
}
fn placeRecord(i:u32) {
  let at=4u+i*6u;
  let slot=slots[u32(shaped[at])];
  if (slot==0xfffffffeu) { return; }
  let ordinal=info[0]+i;
  var lo=0u;
  var hi=scratch[0];
  while (lo<hi) {
    let middle=(lo+hi)/2u;
    if (scratch[4u+params[0]*4u+middle*8u+1u]<=ordinal) { lo=middle+1u; }
    else { hi=middle; }
  }
  if (lo<scratch[0]) {
    let line=4u+params[0]*4u+lo*8u;
    if (ordinal<scratch[line] || ordinal>=scratch[line+1u]) { return; }
    var align=0.0;
    if (params[13]!=0u) {
      align=max(0.0,bitcast<f32>(params[6])-bitcast<f32>(scratch[line+4u]));
      if (params[13]==1u) { align*=0.5; }
    }
    let p=4u+i*4u;
    let scale=bitcast<f32>(params[9]);
    let baseline=fma(f32(info[133]+scratch[line+2u]),bitcast<f32>(params[7]),f32(params[3])*bitcast<f32>(params[8]));
    let x=fma(f32(bitcast<i32>(scratch[p])-bitcast<i32>(scratch[line+3u])+shaped[at+4u]),scale,align);
    let y=fma(f32(bitcast<i32>(scratch[p+1u])+shaped[at+5u]),scale,-baseline);
    let dst=scratch[p+2u]*3u;
    instances[dst]=bitcast<u32>(x); instances[dst+1u]=bitcast<u32>(y); instances[dst+2u]=slot;
    return;
  }
}
@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) gid:vec3u, @builtin(num_workgroups) groups:vec3u) {
  if (carry[0]!=0u || info[140]==0u) { return; }
  let stride=groups.x*groups.y*64u;
  for (var i=gid.x+gid.y*groups.x*64u;i<info[128];i+=stride) { placeRecord(i); }
}
`;

/** Atomic publication is a separate ordered pass after all chunks have placed. */
export const COMPUTE_TEXT_CHUNK_PUBLISH = `
@group(0) @binding(0) var<storage,read> info:array<u32>;
@group(0) @binding(1) var<storage,read> carry:array<u32>;
@group(0) @binding(2) var<storage,read_write> indirect:array<u32>;
@group(0) @binding(3) var<storage,read_write> status:array<u32>;
@compute @workgroup_size(1)
fn main() {
  indirect[0]=6u; indirect[1]=select(0u,info[129],carry[0]==0u); indirect[2]=0u; indirect[3]=0u;
  status[0]=carry[0];
}
`;
