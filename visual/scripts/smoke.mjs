const exercises=["smith_squat","incline_smith_press","lateral_raise"];
const avatars=["soldier","michelle"];
if(exercises.length<3||avatars.length<2) process.exit(1);
console.log(JSON.stringify({ok:true,avatars,exercises}));
