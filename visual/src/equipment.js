import * as THREE from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {ASSET_PATHS} from "./assets.js";

const MAT={
  steel:new THREE.MeshStandardMaterial({color:0x676d73,metalness:.82,roughness:.25}),
  pad:new THREE.MeshStandardMaterial({color:0x22252a,roughness:.9}),
  rubber:new THREE.MeshStandardMaterial({color:0x0b0c0d,roughness:.96})
};

const group=(parent)=>{const g=new THREE.Group();parent.add(g);return g};
const box=(g,w,h,d,x,y,z,mat=MAT.steel,rx=0)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.set(x,y,z);m.rotation.x=rx;m.castShadow=true;m.receiveShadow=true;g.add(m);return m};
const anchor=(g,x,y,z,rx=0)=>{const a=new THREE.Object3D();a.position.set(x,y,z);a.rotation.x=rx;g.add(a);return a};

function setShadows(root){
  root.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true}});
}
function centerRigid(root){
  root.updateMatrixWorld(true);
  const b=new THREE.Box3().setFromObject(root),c=b.getCenter(new THREE.Vector3());
  root.position.sub(c);root.updateMatrixWorld(true);
}
function centerAndFloor(root){
  root.updateMatrixWorld(true);
  const b=new THREE.Box3().setFromObject(root),c=b.getCenter(new THREE.Vector3());
  root.position.x-=c.x;root.position.z-=c.z;root.position.y-=b.min.y;root.updateMatrixWorld(true);
}
async function loadGLB(loader,url,parent,{rigid=false,position=null,rotationY=0}={}){
  const gltf=await loader.loadAsync(url);
  const obj=gltf.scene;
  setShadows(obj);
  if(rigid) centerRigid(obj); else centerAndFloor(obj);
  if(position)obj.position.add(position);
  obj.rotation.y+=rotationY;
  parent.add(obj);
  return {scene:obj,animations:gltf.animations};
}
function fallbackBench(g,incline=0){
  box(g,.58,.11,.55,0,.48,.15,MAT.pad);
  box(g,.58,.11,1.15,0,.90,.48,MAT.pad,THREE.MathUtils.degToRad(-incline));
}
function smithRails(g){
  box(g,.07,2.55,.07,-.93,1.28,.05);
  box(g,.07,2.55,.07,.93,1.28,.05);
  box(g,1.93,.07,.07,0,2.54,.05);
}
function fallbackOHP(g){
  box(g,.65,.12,.55,0,.55,.2,MAT.pad);box(g,.65,.12,1.0,0,1.05,.45,MAT.pad,THREE.MathUtils.degToRad(-12));
  box(g,.12,1.7,.12,-.78,1.05,.15);box(g,.12,1.7,.12,.78,1.05,.15);box(g,1.7,.12,.12,0,1.9,.15);
}
function fallbackTRow(g){
  box(g,.65,.12,.95,0,1.05,.15,MAT.pad,THREE.MathUtils.degToRad(-35));box(g,.75,.1,.5,0,.55,.15,MAT.pad);box(g,.12,1.35,.12,0,.7,-.35);
}

function station(scene,id){
  const root=group(scene),visual=group(root),constraints=group(root);
  root.name=`station:${id}`;
  return {id,group:root,visual,constraints};
}
function control(parent,name){
  const c=group(parent);c.name=`control:${name}`;return c;
}

export function createCommercialGym(scene){
  const S={};

  S.smith=station(scene,"smith");
  smithRails(S.smith.visual);
  S.smith.bar=control(S.smith.constraints,"bar");
  S.smith.bar.position.set(0,1.55,.05);

  S.flat_bench_db=station(scene,"flat_bench_db");
  S.flat_bench_db.bodyAnchor=anchor(S.flat_bench_db.constraints,0,.63,.05,-Math.PI/2);
  S.flat_bench_db.left=control(S.flat_bench_db.constraints,"left-dumbbell");
  S.flat_bench_db.right=control(S.flat_bench_db.constraints,"right-dumbbell");

  S.incline_bench_db=station(scene,"incline_bench_db");
  S.incline_bench_db.bodyAnchor=anchor(S.incline_bench_db.constraints,0,.73,.12,THREE.MathUtils.degToRad(-45));
  S.incline_bench_db.left=control(S.incline_bench_db.constraints,"left-dumbbell");
  S.incline_bench_db.right=control(S.incline_bench_db.constraints,"right-dumbbell");

  S.flat_bench_barbell=station(scene,"flat_bench_barbell");
  S.flat_bench_barbell.bodyAnchor=anchor(S.flat_bench_barbell.constraints,0,.63,.05,-Math.PI/2);
  S.flat_bench_barbell.bar=control(S.flat_bench_barbell.constraints,"bar");

  S.incline_smith=station(scene,"incline_smith");
  S.incline_smith.bodyAnchor=anchor(S.incline_smith.constraints,0,.73,.12,THREE.MathUtils.degToRad(-45));
  smithRails(S.incline_smith.visual);
  S.incline_smith.bar=control(S.incline_smith.constraints,"bar");

  S.ohp_machine=station(scene,"ohp_machine");
  fallbackOHP(S.ohp_machine.visual);
  S.ohp_machine.bodyAnchor=anchor(S.ohp_machine.constraints,0,.72,.20);
  S.ohp_machine.left=control(S.ohp_machine.constraints,"left-handle");
  S.ohp_machine.right=control(S.ohp_machine.constraints,"right-handle");

  S.leg_press=station(scene,"leg_press");
  S.leg_press.bodyAnchor=anchor(S.leg_press.constraints,0,.76,.52,THREE.MathUtils.degToRad(-18));
  S.leg_press.sled=control(S.leg_press.constraints,"sled");
  const plate=box(S.leg_press.sled,1.15,.09,.72,0,0,0,MAT.steel,THREE.MathUtils.degToRad(-45));
  plate.name="kinematic-footplate";

  S.dumbbells=station(scene,"dumbbells");
  S.dumbbells.left=control(S.dumbbells.constraints,"left-dumbbell");
  S.dumbbells.right=control(S.dumbbells.constraints,"right-dumbbell");

  S.lat_pulldown=station(scene,"lat_pulldown");
  S.lat_pulldown.bodyAnchor=anchor(S.lat_pulldown.constraints,0,.69,.15);
  S.lat_pulldown.bar=control(S.lat_pulldown.constraints,"bar");
  box(S.lat_pulldown.bar,1.15,.035,.035,0,0,0);

  S.t_row=station(scene,"t_row");
  fallbackTRow(S.t_row.visual);
  S.t_row.bodyAnchor=anchor(S.t_row.constraints,0,.76,.34,THREE.MathUtils.degToRad(35));
  S.t_row.handle=control(S.t_row.constraints,"handle");
  box(S.t_row.handle,1.05,.04,.04,0,0,0);

  Object.values(S).forEach(s=>s.group.visible=false);
  return S;
}

async function attachRigid(loader,url,controlNode){
  const {scene}=await loadGLB(loader,url,controlNode,{rigid:true});
  return scene;
}
async function attachStatic(loader,url,stationNode){
  const {scene}=await loadGLB(loader,url,stationNode.visual);
  return scene;
}

export async function hydrateCommercialGym(stations){
  const loader=new GLTFLoader();
  const jobs=[
    attachStatic(loader,ASSET_PATHS.gym.powerRack,stations.smith),
    attachStatic(loader,ASSET_PATHS.gym.flatBench,stations.flat_bench_db),
    attachStatic(loader,ASSET_PATHS.gym.adjustableBenchIncline,stations.incline_bench_db),
    attachStatic(loader,ASSET_PATHS.gym.flatBench,stations.flat_bench_barbell),
    attachStatic(loader,ASSET_PATHS.gym.adjustableBenchIncline,stations.incline_smith),
    attachStatic(loader,ASSET_PATHS.gym.legPress,stations.leg_press),
    attachStatic(loader,ASSET_PATHS.gym.latPulldown,stations.lat_pulldown),

    attachRigid(loader,ASSET_PATHS.gym.barbellBare,stations.smith.bar),
    attachRigid(loader,ASSET_PATHS.gym.barbellBare,stations.flat_bench_barbell.bar),
    attachRigid(loader,ASSET_PATHS.gym.barbellBare,stations.incline_smith.bar),

    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.flat_bench_db.left),
    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.flat_bench_db.right),
    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.incline_bench_db.left),
    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.incline_bench_db.right),
    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.dumbbells.left),
    attachRigid(loader,ASSET_PATHS.gym.dumbbell,stations.dumbbells.right)
  ];
  const results=await Promise.allSettled(jobs);
  const failures=results.filter(x=>x.status==="rejected");
  if(failures.length) console.warn("Some real gym assets failed to hydrate",failures.map(x=>x.reason));
  return {loaded:results.length-failures.length,failed:failures.length};
}


export async function loadGymBackdrop(scene){
  const root=new THREE.Group();
  root.name="commercial-gym-backdrop";
  scene.add(root);
  const loader=new GLTFLoader();
  const {scene:obj}=await loadGLB(loader,ASSET_PATHS.gym.studio,root);
  obj.scale.setScalar(.72);
  obj.updateMatrixWorld(true);
  const b=new THREE.Box3().setFromObject(obj);
  const c=b.getCenter(new THREE.Vector3());
  obj.position.x-=c.x;
  obj.position.z-=c.z+7.5;
  obj.position.y-=b.min.y;
  obj.traverse(o=>{
    if(o.isMesh&&o.material){
      const mats=Array.isArray(o.material)?o.material:[o.material];
      for(const m of mats){m.roughness=Math.max(.35,m.roughness??.5);}
    }
  });
  return root;
}
