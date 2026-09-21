export function collectGroundTruth(model,rig,meta){
  const joints={};
  for(const [name,b] of Object.entries(rig)){
    if(!b) continue;
    const v=b.getWorldPosition({x:0,y:0,z:0,set(){return this}});
    joints[name]=[v.x,v.y,v.z];
  }
  return {
    schema_version:1,
    timestamp_ms:performance.now(),
    avatar_id:meta.avatar_id,
    exercise_id:meta.exercise_id,
    form_id:meta.form_id,
    phase:meta.phase,
    issues:meta.issues,
    joints3d:joints
  };
}
