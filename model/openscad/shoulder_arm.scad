include <servo_toothed_head.scad>

module shoulder_arm(){
height = 3.5;
length=45;
$fn=60;
difference(){
union(){
    translate([0,0,0])cylinder(d=10,h=6);

hull(){
translate([0,0,0])cylinder(d=10,h=height);
translate([length,0,0])cylinder(d=8,h=height);
}
}
translate([0,0,2])servo_toothed_head();
translate([0,0,-1])cylinder(d=3.1,h=height*5); // hole for servo
translate([length,0,-1])cylinder(d=3.2,h=height*5); // hole m3


hull(){
translate([8,0,-1])cylinder(d=5.5,h=height*3);
translate([17,0,-1])cylinder(d=4,h=height*3);
  }
hull(){
translate([24,0,-1])cylinder(d=4,h=height*3);
translate([37,0,-1])cylinder(d=4,h=height*3);
  }
}
}

module shoulder_for_mujoco(){
  rotate([90,90,180])shoulder_arm();  
  }
  module shoulder_right_for_mujoco(){
  rotate([0,0,180])shoulder_for_mujoco();
  }
//shoulder_right_for_mujoco();

//translate([length,0,0])cylinder(d=1,h=height);
  
  shoulder_right_for_mujoco();