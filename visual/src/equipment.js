import * as THREE from "three";

const MAT={
  steel:new THREE.MeshStandardMaterial({color:0x6e7378,metalness:.8,roughness:.28}),
  black:new THREE.MeshStandardMaterial({color:0x151719,roughness:.72}),
  pad:new THREE.MeshStandardMaterial({color:0x24272b,roughness:.92}),
  rubber:new THREE.MeshStandardMaterial({color:0x0d0e10,roughness:.96}),
  accent:new THREE.MeshStandardMaterial({color:0x8b9197,metalness:.45,roughness:.42})
};

const addBox=(g,w,h,d,x,y,z,mat=MAT.steel,rx=0,ry=0,rz=0)=>{
  const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat); m.position.set(x,y,z); m.rotation.set(rx,ry,rz); m.castShadow=true; m.receiveShadow=true; g.add(m); return m;
};
const addCyl=(g,r,depth,x,y,z,axis="x",mat=MAT.rubber)=>{
  const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,depth,28),mat);
  if(axis==="x") m.rotation.z=Math.PI/2; else if(axis==="z") m.rotation.x=Math.PI/2;
  m.position.set(x,y,z); m.castShadow=true; g.add(m); return m;
};
const group=(scene)=>{const g=new THREE.Group(); scene.add(g); return g};

function bench(g,{incline=0,x=0,y=.48,z=.15}={}){
  const seat=addBox(g,.58,.11,.55,x,y,z,MAT.pad);
  const back=addBox(g,.58,.11,1.15,x,y+.42,z+.33,MAT.pad,THREE.MathUtils.degToRad(-incline),0,0);
  addBox(g,.08,.55,.08,x-.23,y-.30,z+.1,MAT.steel); addBox(g,.08,.55,.08,x+.23,y-.30,z+.1,MAT.steel);
  return {seat,back};
}
function dumbbell(g,x,y,z){
  const handle=addBox(g,.26,.035,.035,x,y,z,MAT.steel);
  addCyl(g,.095,.07,x-.16,y,z,"x"); addCyl(g,.095,.07,x+.16,y,z,"x");
  return handle;
}
function barbell(g,y=1.2,z=0){
  const bar=addBox(g,2.25,.045,.045,0,y,z,MAT.steel);
  for(const x of [-1.12,-1.22,1.12,1.22]) addCyl(g,.22,.075,x,y,z,"x");
  return bar;
}

export function createCommercialGym(scene){
  const stations={};

  {
    const g=group(scene);
    addBox(g,.12,2.95,.12,-.98,1.475,0); addBox(g,.12,2.95,.12,.98,1.475,0); addBox(g,2.08,.12,.12,0,2.9,0);
    const bar=barbell(g,1.8,0);
    stations.smith={group:g,bar};
  }

  {
    const g=group(scene); bench(g,{incline:0,z:.25}); const left=dumbbell(g,-.48,1.02,.05), right=dumbbell(g,.48,1.02,.05);
    stations.flat_bench_db={group:g,left,right};
  }

  {
    const g=group(scene); bench(g,{incline:32,z:.2}); const left=dumbbell(g,-.48,1.1,.02), right=dumbbell(g,.48,1.1,.02);
    stations.incline_bench_db={group:g,left,right};
  }

  {
    const g=group(scene); bench(g,{incline:0,z:.25}); const bar=barbell(g,1.15,-.05);
    addBox(g,.1,1.6,.1,-.85,.8,-.05); addBox(g,.1,1.6,.1,.85,.8,-.05);
    stations.flat_bench_barbell={group:g,bar};
  }

  {
    const g=group(scene); bench(g,{incline:32,z:.22}); addBox(g,.12,2.95,.12,-.98,1.475,0); addBox(g,.12,2.95,.12,.98,1.475,0); addBox(g,2.08,.12,.12,0,2.9,0);
    const bar=barbell(g,1.45,-.02); stations.incline_smith={group:g,bar};
  }

  {
    const g=group(scene); addBox(g,.65,.12,.55,0,.55,.2,MAT.pad); addBox(g,.65,.12,1.0,0,1.05,.45,MAT.pad,THREE.MathUtils.degToRad(-12));
    addBox(g,1.7,.12,.12,0,1.95,.15); addBox(g,.12,1.7,.12,-.78,1.05,.15); addBox(g,.12,1.7,.12,.78,1.05,.15);
    const left=addBox(g,.45,.06,.06,-.58,1.55,.05), right=addBox(g,.45,.06,.06,.58,1.55,.05);
    stations.ohp_machine={group:g,left,right};
  }

  {
    const g=group(scene);
    addBox(g,1.0,.18,1.55,0,.62,.25,MAT.pad,THREE.MathUtils.degToRad(-42));
    addBox(g,1.3,.12,.12,0,.3,-.35); addBox(g,.12,1.65,.12,-.55,1.05,-.25); addBox(g,.12,1.65,.12,.55,1.05,-.25);
    const sled=addBox(g,1.15,.10,.75,0,1.28,-.05,MAT.steel,THREE.MathUtils.degToRad(-42));
    stations.leg_press={group:g,sled};
  }

  {
    const g=group(scene); addBox(g,.95,.12,.55,0,.55,.15,MAT.pad); addBox(g,.12,2.6,.12,-.65,1.3,0); addBox(g,.12,2.6,.12,.65,1.3,0); addBox(g,1.42,.1,.1,0,2.55,0);
    const bar=addBox(g,1.15,.05,.05,0,2.05,.1,MAT.steel); addBox(g,.02,.6,.02,0,2.35,.1,MAT.steel);
    stations.lat_pulldown={group:g,bar};
  }

  {
    const g=group(scene); addBox(g,.65,.12,.95,0,1.05,.15,MAT.pad,THREE.MathUtils.degToRad(-20)); addBox(g,.75,.1,.5,0,.55,.15,MAT.pad);
    addBox(g,.12,1.35,.12,0,.7,-.35); const handle=addBox(g,1.05,.055,.055,0,.78,-.65,MAT.steel);
    stations.t_row={group:g,handle};
  }

  {
    const g=group(scene); const left=dumbbell(g,-.5,1.0,0),right=dumbbell(g,.5,1.0,0); stations.dumbbells={group:g,left,right};
  }

  for(const s of Object.values(stations)) s.group.visible=false;
  return stations;
}
