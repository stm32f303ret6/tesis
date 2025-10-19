include <servo.scad>
width = 6; // width of one side in Y axis
tilt_screw_separation = 5;// tilt servo screw distance separation
bearing_distance = 0.1+3.5; // 0.1 lo minimo
heigth = 8;
m3=3.2;
bearing_diameter=6.05;

module servo_mount(){
  tolerance = 0.1;
  cube([40.0+tolerance, 20.0+tolerance, 41.0+tolerance], center=true);
  }

module two_servo_base(){
translate([10,0,0])
difference(){
  translate([-10,0,0])cube([54,42+width*2,heigth], center=true);
  
color([1,0,0,0.2])
union(){
translate([-10,-11,0])servo_mount();
translate([-10,11,0])servo_mount();  
}


// mr63zz bearing
translate([-10,41+bearing_distance,0])rotate([90,0,0])cylinder(d=bearing_diameter,h=20,$fn=60);
// bearing screw 3mm
translate([-10,40,0])rotate([90,0,0])cylinder(d=m3,h=20,$fn=20);

translate([-10,21.1+2,0])rotate([90,0,0])cylinder(d=m3+3,h=12,$fn=20);


// arm servo screws
translate([-34.5,6,-10])cylinder(d=m3,h=20,$fn=20);
translate([-34.5,16,-10])cylinder(d=m3,h=20,$fn=20);
translate([-34.5,-6,-10])cylinder(d=m3,h=20,$fn=20);
translate([-34.5,-16,-10])cylinder(d=m3,h=20,$fn=20);

translate([14.5,6,-10])cylinder(d=m3,h=20,$fn=20);
translate([14.5,16,-10])cylinder(d=m3,h=20,$fn=20);
translate([14.5,-6,-10])cylinder(d=m3,h=20,$fn=20);
translate([14.5,-16,-10])cylinder(d=m3,h=20,$fn=20);
// tilt servo screws
translate([-10 + tilt_screw_separation,-10,0])rotate([90,0,0])cylinder(d=m3,h=20,$fn=20);
translate([-10 - tilt_screw_separation,-10,0])rotate([90,0,0])cylinder(d=m3,h=20,$fn=20);

translate([-10 + tilt_screw_separation,-10,0])rotate([90,0,0])cylinder(d=m3+3,h=11.1+2,$fn=20);
translate([-10 - tilt_screw_separation,-10,0])rotate([90,0,0])cylinder(d=m3+3,h=11.1+2,$fn=20);

}

}

module two_servo_base_for_mujoco(){
color([0,0.8,0])
rotate([90,90,0])
two_servo_base();
}

module two_servo_for_mujoco(){
 translate([11,-15,-10])servo_for_mujoco();
 translate([-11,-15,-10])servo_for_mujoco();
}

