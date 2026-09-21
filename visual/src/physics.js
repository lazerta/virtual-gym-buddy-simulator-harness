import * as THREE from "three";
import RAPIER from "@dimforge/rapier3d-deterministic-compat";

const v3=v=>({x:v.x,y:v.y,z:v.z});
const q4=q=>({x:q.x,y:q.y,z:q.z,w:q.w});

function worldPose(object){
  object.updateWorldMatrix(true,false);
  return {
    position:object.getWorldPosition(new THREE.Vector3()),
    quaternion:object.getWorldQuaternion(new THREE.Quaternion())
  };
}

function setObjectWorldPose(object,position,quaternion){
  const parent=object.parent;
  const localPos=position.clone();
  if(parent){
    parent.updateWorldMatrix(true,false);
    parent.worldToLocal(localPos);
    const parentQ=parent.getWorldQuaternion(new THREE.Quaternion()).invert();
    object.quaternion.copy(parentQ.multiply(quaternion.clone()));
  }else{
    object.quaternion.copy(quaternion);
  }
  object.position.copy(localPos);
  object.updateMatrixWorld(true);
}

export class GymPhysics{
  static async create(){
    await RAPIER.init();
    return new GymPhysics();
  }

  constructor(){
    this.world=new RAPIER.World({x:0,y:-9.81,z:0});
    this.world.timestep=1/120;
    this.entries=new Map();
    const floor=this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0,-.06,0));
    this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(12,.06,12).setFriction(.85).setRestitution(0),
      floor
    );
  }

  registerDynamicTracked(key,object,{mass=10,halfExtents=[.18,.08,.08],linearDamping=4,angularDamping=5,kp=220,kd=28}={}){
    if(this.entries.has(key))return this.entries.get(key);
    const pose=worldPose(object);
    const body=this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(pose.position.x,pose.position.y,pose.position.z)
        .setLinearDamping(linearDamping)
        .setAngularDamping(angularDamping)
        .setCcdEnabled(true)
    );
    body.setRotation(q4(pose.quaternion),true);
    const collider=this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(...halfExtents).setMass(mass).setFriction(.65).setRestitution(.02),
      body
    );
    const entry={key,type:"dynamic",object,body,collider,mass,kp,kd,target:pose.position.clone(),targetQ:pose.quaternion.clone(),error:0};
    this.entries.set(key,entry);
    return entry;
  }

  registerKinematic(key,object,{halfExtents=[.25,.05,.05]}={}){
    if(this.entries.has(key))return this.entries.get(key);
    const pose=worldPose(object);
    const body=this.world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(pose.position.x,pose.position.y,pose.position.z)
    );
    body.setRotation(q4(pose.quaternion),true);
    const collider=this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(...halfExtents).setFriction(.7).setRestitution(0),
      body
    );
    const entry={key,type:"kinematic",object,body,collider,target:pose.position.clone(),targetQ:pose.quaternion.clone(),error:0};
    this.entries.set(key,entry);
    return entry;
  }

  captureTarget(key){
    const e=this.entries.get(key);
    if(!e)return;
    const pose=worldPose(e.object);
    e.target.copy(pose.position);
    e.targetQ.copy(pose.quaternion);
  }

  captureAllTargets(){
    for(const key of this.entries.keys())this.captureTarget(key);
  }

  resetToTargets(){
    for(const e of this.entries.values()){
      e.body.setTranslation(v3(e.target),true);
      e.body.setRotation(q4(e.targetQ),true);
      e.body.setLinvel({x:0,y:0,z:0},true);
      e.body.setAngvel({x:0,y:0,z:0},true);
      e.error=0;
    }
    this.world.propagateModifiedBodyPositionsToColliders();
    this.syncVisuals();
  }

  _applyDynamicTracking(e){
    const p=e.body.translation(),vel=e.body.linvel();
    const dx=e.target.x-p.x,dy=e.target.y-p.y,dz=e.target.z-p.z;
    const force={
      x:e.mass*(e.kp*dx-e.kd*vel.x),
      y:e.mass*(e.kp*dy-e.kd*vel.y)+e.mass*9.81,
      z:e.mass*(e.kp*dz-e.kd*vel.z)
    };
    e.body.addForce(force,true);

    const q=e.body.rotation();
    const current=new THREE.Quaternion(q.x,q.y,q.z,q.w);
    const inv=current.clone().invert();
    const dq=e.targetQ.clone().multiply(inv).normalize();
    const angle=2*Math.acos(Math.min(1,Math.abs(dq.w)));
    if(angle>1e-4){
      const s=Math.sqrt(Math.max(1e-8,1-dq.w*dq.w));
      const axis=new THREE.Vector3(dq.x/s,dq.y/s,dq.z/s).normalize();
      const av=e.body.angvel();
      const torqueScale=e.mass*35*angle;
      e.body.addTorque({
        x:axis.x*torqueScale-av.x*e.mass*4,
        y:axis.y*torqueScale-av.y*e.mass*4,
        z:axis.z*torqueScale-av.z*e.mass*4
      },true);
    }
  }

  step(substeps=2){
    const n=Math.max(1,Math.min(8,substeps|0));
    this.world.timestep=1/(60*n);
    for(let i=0;i<n;i++){
      for(const e of this.entries.values()){
        if(e.type==="kinematic"){
          e.body.setNextKinematicTranslation(v3(e.target));
          e.body.setNextKinematicRotation(q4(e.targetQ));
        }else{
          this._applyDynamicTracking(e);
        }
      }
      this.world.step();
    }
    for(const e of this.entries.values()){
      const p=e.body.translation();
      e.error=Math.hypot(e.target.x-p.x,e.target.y-p.y,e.target.z-p.z);
    }
    this.syncVisuals();
  }

  syncVisuals(){
    for(const e of this.entries.values()){
      const p=e.body.translation(),q=e.body.rotation();
      setObjectWorldPose(
        e.object,
        new THREE.Vector3(p.x,p.y,p.z),
        new THREE.Quaternion(q.x,q.y,q.z,q.w)
      );
    }
  }

  metrics(){
    const out={};
    for(const [key,e] of this.entries)out[key]={type:e.type,error_m:e.error};
    return out;
  }

  maxError(){
    let m=0;
    for(const e of this.entries.values())m=Math.max(m,e.error||0);
    return m;
  }
}

export function registerGymPhysics(physics,gym){
  physics.registerKinematic("smith.bar",gym.smith.bar,{halfExtents:[1.1,.035,.035]});
  physics.registerKinematic("incline_smith.bar",gym.incline_smith.bar,{halfExtents:[1.1,.035,.035]});
  physics.registerKinematic("leg_press.sled",gym.leg_press.sled,{halfExtents:[.58,.08,.36]});
  physics.registerKinematic("lat_pulldown.bar",gym.lat_pulldown.bar,{halfExtents:[.58,.03,.03]});
  physics.registerKinematic("ohp.left",gym.ohp_machine.left,{halfExtents:[.20,.04,.04]});
  physics.registerKinematic("ohp.right",gym.ohp_machine.right,{halfExtents:[.20,.04,.04]});
  physics.registerKinematic("trow.handle",gym.t_row.handle,{halfExtents:[.52,.03,.03]});

  physics.registerDynamicTracked("flat_barbell.bar",gym.flat_bench_barbell.bar,{mass:20,halfExtents:[1.12,.04,.04],kp:260,kd:34});
  physics.registerDynamicTracked("flat_db.left",gym.flat_bench_db.left,{mass:10,halfExtents:[.18,.10,.10]});
  physics.registerDynamicTracked("flat_db.right",gym.flat_bench_db.right,{mass:10,halfExtents:[.18,.10,.10]});
  physics.registerDynamicTracked("incline_db.left",gym.incline_bench_db.left,{mass:10,halfExtents:[.18,.10,.10]});
  physics.registerDynamicTracked("incline_db.right",gym.incline_bench_db.right,{mass:10,halfExtents:[.18,.10,.10]});
  physics.registerDynamicTracked("raise_db.left",gym.dumbbells.left,{mass:7.5,halfExtents:[.16,.09,.09]});
  physics.registerDynamicTracked("raise_db.right",gym.dumbbells.right,{mass:7.5,halfExtents:[.16,.09,.09]});
}
