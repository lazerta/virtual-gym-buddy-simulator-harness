import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
import {AVATARS,EXERCISES,FORMS} from "./catalog.js";
import {resolveRig,captureRest,captureRootRest,normalizeAvatar,measureRig} from "./rig.js";
import {createCommercialGym,hydrateCommercialGym,loadGymBackdrop} from "./equipment.js";
import {applyMotion,cyclePhase} from "./motion.js";
import {collectGroundTruth} from "./groundTruth.js";
import {loadMotionPriors} from "./motionPrior.js";
import {GymPhysics,registerGymPhysics} from "./physics.js";

const app=document.getElementById("app"),hud=document.getElementById("hud");
const avatarSel=document.getElementById("avatar"),exerciseSel=document.getElementById("exercise"),formSel=document.getElementById("form");
Object.values(AVATARS).forEach(a=>avatarSel.add(new Option(a.label,a.id)));
Object.values(EXERCISES).forEach(e=>exerciseSel.add(new Option(e.label,e.id)));
Object.values(FORMS).forEach(f=>formSel.add(new Option(f.label,f.id)));

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x202226);
scene.fog=new THREE.Fog(0x202226,14,34);

const camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,.05,100);
camera.position.set(4.2,2.6,5.4);

const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance",preserveDrawingBuffer:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled=true;
renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.05;
app.appendChild(renderer.domElement);

const controls=new OrbitControls(camera,renderer.domElement);
controls.target.set(0,1,0);
controls.enableDamping=true;
controls.maxPolarAngle=Math.PI*.52;
controls.minDistance=2.2;
controls.maxDistance=10;

scene.add(new THREE.HemisphereLight(0xffffff,0x30343a,1.4));
const sun=new THREE.DirectionalLight(0xffffff,3.5);
sun.position.set(4,8,3);
sun.castShadow=true;
sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.near=.1;sun.shadow.camera.far=30;
sun.shadow.camera.left=-7;sun.shadow.camera.right=7;sun.shadow.camera.top=7;sun.shadow.camera.bottom=-7;
scene.add(sun);

const floor=new THREE.Mesh(new THREE.PlaneGeometry(24,24),new THREE.MeshStandardMaterial({color:0x303336,roughness:.96}));
floor.rotation.x=-Math.PI/2;
floor.receiveShadow=true;
scene.add(floor);

const gym=createCommercialGym(scene);
const loader=new GLTFLoader();

let model=null,rig={},rigRest={},rootRest=null,metrics=null,generation=0,currentEquipment=null;
let playing=true,manualPhase=0,lastTruth=null;
let physics=null,motionPriors={squat:null};
const clock=new THREE.Clock();
let assetHealth={realEquipment:0,failedEquipment:0,backdrop:false};

function configureEquipment(){
  Object.values(gym).forEach(s=>s.group.visible=false);
  const key=EXERCISES[exerciseSel.value]?.equipment;
  currentEquipment=gym[key];
  if(!currentEquipment)throw new Error(`missing equipment station: ${key}`);
  currentEquipment.group.visible=true;
}

async function loadAvatar(id=avatarSel.value){
  const spec=AVATARS[id];
  if(!spec)throw new Error(`unknown avatar: ${id}`);
  const gen=++generation;
  const gltf=await loader.loadAsync(spec.url);
  if(gen!==generation)return;
  if(model)scene.remove(model);
  model=gltf.scene;
  normalizeAvatar(model,spec.targetHeight);
  model.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true}});
  scene.add(model);
  rig=resolveRig(model);
  const required=["hips","leftUpperArm","leftLowerArm","leftHand","rightUpperArm","rightLowerArm","rightHand","leftUpperLeg","leftLowerLeg","leftFoot","rightUpperLeg","rightLowerLeg","rightFoot"];
  const missing=required.filter(k=>!rig[k]);
  if(missing.length)throw new Error(`avatar ${id} rig missing: ${missing.join(", ")}`);
  rigRest=captureRest(rig);
  rootRest=captureRootRest(model);
  metrics=measureRig(model,rig);
  return metrics;
}

function updateHUD(gt){
  const inFrame=Object.values(gt.joints2d).filter(x=>x.in_frame).length;
  const physicsErr=gt.physics?.max_error_m??0;
  const prior=gt.motion_prior?.source??"constraint";
  hud.textContent=`Gym Buddy Hybrid 3D Harness
avatar: ${gt.avatar_id} (${metrics.height.toFixed(2)} m)
exercise: ${gt.exercise_id}
form: ${gt.form_id}
phase: ${gt.phase.toFixed(3)}
human IK error: ${(gt.constraint_error_m*100).toFixed(1)} cm
equipment physics error: ${(physicsErr*100).toFixed(1)} cm
motion source: ${prior}
joints: ${Object.keys(gt.joints3d).length}  in-frame: ${inFrame}
L elbow: ${gt.joint_angles.left_elbow_flex_deg?.toFixed(1)??"n/a"}°
L knee: ${gt.joint_angles.left_knee_flex_deg?.toFixed(1)??"n/a"}°
real assets: ${assetHealth.realEquipment}  failed: ${assetHealth.failedEquipment}
issues: ${JSON.stringify(gt.issues)}`;
}

function solveTargetsAtPhase(phase){
  const p=Math.max(0,Math.min(1,Number(phase)||0));
  return applyMotion({
    exercise:exerciseSel.value,
    form:formSel.value,
    phase:p,
    model,rig,rigRest,rootRest,metrics,
    station:currentEquipment,
    motionPriors
  });
}

function renderAtPhase(phase,{resetPhysics=false}={}){
  if(!model||!metrics||!currentEquipment||!physics)return null;

  const truth=solveTargetsAtPhase(phase);

  physics.captureAllTargets();
  if(resetPhysics)physics.resetToTargets();
  else physics.step(2);

  const physicsState=physics.metrics();
  const gt=collectGroundTruth(model,rig,{
    avatar_id:avatarSel.value,
    exercise_id:exerciseSel.value,
    form_id:formSel.value,
    phase:truth.phase,
    issues:truth.issues,
    constraint_error_m:truth.constraint_error_m,
    measured:truth.measured,
    motion_prior:truth.motion_prior??null,
    physics:{
      max_error_m:physics.maxError(),
      bodies:physicsState
    }
  },camera,renderer);

  window.__GYM_BUDDY_GT__=gt;
  lastTruth=gt;
  updateHUD(gt);
  controls.update();
  renderer.render(scene,camera);
  return gt;
}

async function setScenario(spec={}){
  playing=false;
  const avatar=spec.avatar_id??spec.avatar??avatarSel.value;
  if(avatar!==avatarSel.value){
    avatarSel.value=avatar;
    await loadAvatar(avatar);
  }
  if(spec.exercise_id??spec.exercise){
    const id=spec.exercise_id??spec.exercise;
    if(!EXERCISES[id])throw new Error(`unknown exercise: ${id}`);
    exerciseSel.value=id;
  }
  if(spec.form_id??spec.form){
    const id=spec.form_id??spec.form;
    if(!FORMS[id])throw new Error(`unknown form: ${id}`);
    formSel.value=id;
  }
  configureEquipment();
  manualPhase=Math.max(0,Math.min(1,Number(spec.phase??manualPhase)||0));
  return renderAtPhase(manualPhase,{resetPhysics:true});
}

avatarSel.addEventListener("change",async()=>{
  playing=false;
  await loadAvatar();
  renderAtPhase(manualPhase,{resetPhysics:true});
});
exerciseSel.addEventListener("change",()=>{
  playing=false;
  configureEquipment();
  renderAtPhase(manualPhase,{resetPhysics:true});
});
formSel.addEventListener("change",()=>{
  playing=false;
  renderAtPhase(manualPhase,{resetPhysics:true});
});

avatarSel.value="quaternius_human";
exerciseSel.value="smith_squat";
formSel.value="correct";

configureEquipment();

const [hydration,backdropResult,priors,physicsInstance]=await Promise.all([
  hydrateCommercialGym(gym),
  loadGymBackdrop(scene).then(()=>true).catch(e=>{console.warn("gym backdrop failed",e);return false}),
  loadMotionPriors().catch(e=>{console.warn("motion priors failed",e);return {squat:null}}),
  GymPhysics.create()
]);

assetHealth={realEquipment:hydration.loaded,failedEquipment:hydration.failed,backdrop:backdropResult};
motionPriors=priors;
physics=physicsInstance;
registerGymPhysics(physics,gym);

await loadAvatar();

solveTargetsAtPhase(manualPhase);
physics.captureAllTargets();
physics.resetToTargets();

window.__GYM_BUDDY_SET_SCENARIO__=setScenario;
window.__GYM_BUDDY_STEP__=phase=>{
  playing=false;
  manualPhase=Math.max(0,Math.min(1,Number(phase)||0));
  return renderAtPhase(manualPhase,{resetPhysics:false});
};
window.__GYM_BUDDY_RESET_STEP__=phase=>{
  playing=false;
  manualPhase=Math.max(0,Math.min(1,Number(phase)||0));
  return renderAtPhase(manualPhase,{resetPhysics:true});
};
window.__GYM_BUDDY_PLAY__=()=>{playing=true;clock.start();return true};
window.__GYM_BUDDY_PAUSE__=()=>{playing=false;return lastTruth};
window.__GYM_BUDDY_SET_CAMERA__=({position,target,fov}={})=>{
  if(position)camera.position.fromArray(position);
  if(target)controls.target.fromArray(target);
  if(Number.isFinite(fov)){
    camera.fov=Math.max(20,Math.min(100,fov));
    camera.updateProjectionMatrix();
  }
  return renderAtPhase(manualPhase,{resetPhysics:false});
};
window.__GYM_BUDDY_CAPTURE_PNG__=()=>renderer.domElement.toDataURL("image/png");
window.__GYM_BUDDY_READY__=true;

renderer.setAnimationLoop(()=>{
  if(playing)manualPhase=cyclePhase(clock.getElapsedTime());
  renderAtPhase(manualPhase,{resetPhysics:false});
});

addEventListener("resize",()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  renderAtPhase(manualPhase,{resetPhysics:false});
});
