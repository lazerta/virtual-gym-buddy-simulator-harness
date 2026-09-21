import {test,expect} from "@playwright/test";

const AVATARS=["quaternius_human","quaternius_superhero_male"];
const EXERCISES=[
  "incline_db_press","flat_db_press","barbell_bench","incline_smith_press","seated_ohp",
  "smith_squat","leg_press","lateral_raise","lat_pulldown","chest_supported_t_row"
];

test("real GLB simulation runs every canonical exercise",async({page})=>{
  const runtimeErrors=[];
  page.on("pageerror",e=>runtimeErrors.push(String(e)));
  page.on("console",m=>{if(m.type()==="error")runtimeErrors.push(m.text())});

  await page.goto("/");
  await page.waitForFunction(()=>window.__GYM_BUDDY_READY__===true,{timeout:90000});

  for(const avatar of AVATARS){
    for(const exercise of EXERCISES){
      for(const phase of [0,.5,1]){
        const gt=await page.evaluate(async({avatar,exercise,phase})=>{
          return await window.__GYM_BUDDY_SET_SCENARIO__({
            avatar_id:avatar,exercise_id:exercise,form_id:"correct",phase
          });
        },{avatar,exercise,phase});

        expect(gt.avatar_id).toBe(avatar);
        expect(gt.exercise_id).toBe(exercise);
        expect(Object.keys(gt.joints3d).length).toBeGreaterThanOrEqual(13);
        expect(Object.keys(gt.joints2d).length).toBe(Object.keys(gt.joints3d).length);
        expect(Number.isFinite(gt.constraint_error_m)).toBeTruthy();
        expect(gt.constraint_error_m).toBeLessThan(.12);

        for(const xyz of Object.values(gt.joints3d)){
          expect(xyz).toHaveLength(3);
          expect(xyz.every(Number.isFinite)).toBeTruthy();
        }
        for(const p of Object.values(gt.joints2d)){
          expect(Number.isFinite(p.x_px)&&Number.isFinite(p.y_px)).toBeTruthy();
        }
      }
    }
  }

  const png=await page.evaluate(()=>window.__GYM_BUDDY_CAPTURE_PNG__());
  expect(png.startsWith("data:image/png;base64,")).toBeTruthy();
  expect(png.length).toBeGreaterThan(10000);
  expect(runtimeErrors).toEqual([]);
});
