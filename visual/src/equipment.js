import * as THREE from "three";

const MAT={
  steel:new THREE.MeshStandardMaterial({color:0x676d73,metalness:.82,roughness:.25}),
  dark:new THREE.MeshStandardMaterial({color:0x141619,roughness:.8}),
  pad:new THREE.MeshStandardMaterial({color:0x22252a,roughness:.9}),
  rubber:new THREE.MeshStandardMaterial({color:0x0b0c0d,roughness:.96})
};
const addBox=(g,w,h,d,x,y,z,mat=MAT.steel,rx=0)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.set(x,y,z);m.rotation.x=rx;m.castShadow=true;m.receiveShadow=true;g.add(m);return m};
const addCyl=(g,r,depth,x,y,z,axis="x",mat=MAT.rubber)=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,depth,28),mat);if(axis==="x")m.rotation.z=Math.PI/2;else if(axis==="z")m.rotation.x=Math.PI/2;m.position.set(x,y,z);m.castShadow=true;g.add(m);return m};
const anchor=(g,x,y,z,rx=0)=>{const a=new THREE.Object3D();a.position.set(x,y,z);a.rotation.x=rx;g.add(a);return a};
const group=scene=>{const g=new THREE.Group();scene.add(g);return g};
function bench(g,{incline=0,y=.48,z=.15}={}){addBox(g,.58,.11,.55,0,y,z,MAT.pad);addBox(g,.58,.11,1.15,0,y+.42,z+.33,MAT.pad,THREE.MathUtils.degToRad(-incline));addBox(g,.08,.55,.08,-.23,y-.30,z+.1);addBox(g,.08,.55,.08,.23,y-.30,z+.1)}
function dumbbell(g,x,y,z){const h=addBox(g,.26,.035,.035,x,y,z,MAT.steel);addCyl(g,.095,.07,x-.16,y,z);addCyl(g,.095,.07,x+.16,y,z);return h}
function barbell(g,y,z){const bar=addBox(g,2.25,.045,.045,0,y,z);for(const x of[-1.12,-1.22,1.12,1.22])addCyl(g,.22,.075,x,y,z);return bar}

export function createCommercialGym(scene){
  const S={};
  {const g=group(scene);addBox(g,.12,2.95,.12,-.98,1.475,0);addBox(g,.12,2.95,.12,.98,1.475,0);addBox(g,2.08,.12,.12,0,2.9,0);S.smith={group:g,bar:barbell(g,1.8,0)}}
  {const g=group(scene);bench(g,{incline:0});S.flat_bench_db={group:g,bodyAnchor:anchor(g,0,.68,.20,-Math.PI/2),left:dumbbell(g,-.45,1.0,0),right:dumbbell(g,.45,1.0,0)}}
  {const g=group(scene);bench(g,{incline:30});S.incline_bench_db={group:g,bodyAnchor:anchor(g,0,.70,.22,THREE.MathUtils.degToRad(-60)),left:dumbbell(g,-.45,1.05,0),right:dumbbell(g,.45,1.05,0)}}
  {const g=group(scene);bench(g,{incline:0});addBox(g,.1,1.6,.1,-.85,.8,-.05);addBox(g,.1,1.6,.1,.85,.8,-.05);S.flat_bench_barbell={group:g,bodyAnchor:anchor(g,0,.68,.20,-Math.PI/2),bar:barbell(g,1.15,-.05)}}
  {const g=group(scene);bench(g,{incline:30});addBox(g,.12,2.95,.12,-.98,1.475,0);addBox(g,.12,2.95,.12,.98,1.475,0);addBox(g,2.08,.12,.12,0,2.9,0);S.incline_smith={group:g,bodyAnchor:anchor(g,0,.70,.22,THREE.MathUtils.degToRad(-60)),bar:barbell(g,1.45,-.02)}}
  {const g=group(scene);addBox(g,.65,.12,.55,0,.55,.2,MAT.pad);addBox(g,.65,.12,1.0,0,1.05,.45,MAT.pad,THREE.MathUtils.degToRad(-12));addBox(g,1.7,.12,.12,0,1.95,.15);addBox(g,.12,1.7,.12,-.78,1.05,.15);addBox(g,.12,1.7,.12,.78,1.05,.15);S.ohp_machine={group:g,bodyAnchor:anchor(g,0,.75,.20),left:addBox(g,.45,.06,.06,-.58,1.55,.05),right:addBox(g,.45,.06,.06,.58,1.55,.05)}}
  {const g=group(scene);addBox(g,1.0,.18,1.55,0,.62,.25,MAT.pad,THREE.MathUtils.degToRad(-42));addBox(g,1.3,.12,.12,0,.3,-.35);addBox(g,.12,1.65,.12,-.55,1.05,-.25);addBox(g,.12,1.65,.12,.55,1.05,-.25);S.leg_press={group:g,bodyAnchor:anchor(g,0,.78,.52,THREE.MathUtils.degToRad(-18)),sled:addBox(g,1.15,.10,.75,0,1.28,-.05,MAT.steel,THREE.MathUtils.degToRad(-42))}}
  {const g=group(scene);S.dumbbells={group:g,left:dumbbell(g,-.5,1.0,0),right:dumbbell(g,.5,1.0,0)}}
  {const g=group(scene);addBox(g,.95,.12,.55,0,.55,.15,MAT.pad);addBox(g,.12,2.6,.12,-.65,1.3,0);addBox(g,.12,2.6,.12,.65,1.3,0);addBox(g,1.42,.1,.1,0,2.55,0);S.lat_pulldown={group:g,bodyAnchor:anchor(g,0,.72,.18),bar:addBox(g,1.15,.05,.05,0,2.05,.1)}}
  {const g=group(scene);addBox(g,.65,.12,.95,0,1.05,.15,MAT.pad,THREE.MathUtils.degToRad(-35));addBox(g,.75,.1,.5,0,.55,.15,MAT.pad);addBox(g,.12,1.35,.12,0,.7,-.35);S.t_row={group:g,bodyAnchor:anchor(g,0,.78,.38,THREE.MathUtils.degToRad(35)),handle:addBox(g,1.05,.055,.055,0,.78,-.65)}}
  Object.values(S).forEach(s=>s.group.visible=false);return S;
}
