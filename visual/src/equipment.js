import * as THREE from "three";

const steel=()=>new THREE.MeshStandardMaterial({color:0x777777,metalness:.8,roughness:.3});
const dark=()=>new THREE.MeshStandardMaterial({color:0x171717,roughness:.8});

function box(scene,w,h,d,x,y,z,mat=steel()){
  const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);
  m.position.set(x,y,z); m.castShadow=true; scene.add(m); return m;
}

export function createSmithMachine(scene){
  const g=new THREE.Group(); scene.add(g);
  const add=(w,h,d,x,y,z,mat=steel())=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);m.position.set(x,y,z);m.castShadow=true;g.add(m);return m};
  add(.12,2.9,.12,-.95,1.45,0); add(.12,2.9,.12,.95,1.45,0); add(2.05,.12,.12,0,2.88,0);
  const bar=add(2.2,.06,.06,0,1.8,0);
  for(const x of [-1.14,-1.24,1.14,1.24]){
    const p=new THREE.Mesh(new THREE.CylinderGeometry(.22,.22,.08,24),dark());
    p.rotation.z=Math.PI/2; p.position.set(x,1.8,0); g.add(p);
  }
  return {group:g,bar};
}

export function createDumbbells(scene){
  const g=new THREE.Group(); scene.add(g);
  const make=(x)=>{const h=box(g,.28,.05,.05,x,1.1,0,dark());return h};
  return {group:g,left:make(-.45),right:make(.45)};
}
