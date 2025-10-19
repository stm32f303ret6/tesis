
module elbow_arm(){
height = 3.5;
length=60;
$fn=60;
difference(){
union(){

hull(){
translate([0,0,0])cylinder(d=10,h=height);
translate([length,0,0])cylinder(d=10,h=height);
}
}
translate([0,0,-1])cylinder(d=6.12,h=height*5); // hole for bearing
translate([length,0,-1])cylinder(d=6.12,h=height*5); // hole for bearing



// center holes
hull(){
translate([10,0,-1])cylinder(d=6,h=height*3);
translate([21,0,-1])cylinder(d=6,h=height*3);
}
hull(){
translate([length-30,0,-1])cylinder(d=6,h=height*3);
translate([length-10,0,-1])cylinder(d=6,h=height*3);
}

}
}


module elbow_for_mujoco(){
  rotate([90,90,180])elbow_arm();  
  }
  module elbow_right_for_mujoco(){
rotate([0,0,180])  
  elbow_for_mujoco();
  }
//elbow_right_for_mujoco();
  
  elbow_right_for_mujoco();