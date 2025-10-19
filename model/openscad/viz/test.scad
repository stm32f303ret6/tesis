include <../two_servo_base.scad>
include <../servo.scad>
include <../shoulder_arm.scad>
include <../elbow_arm.scad>

rotate([0,90,270]){
color([1,0,0])translate([0,18.05,15])rotate([0,0,270])servo();
color([0,0,1])translate([0,-18.05,15])rotate([0,0,90])servo();
color([0,1,0])translate([0,-65,0])rotate([90,270,180])servo();

}


color([1,1,0])translate([18,-24,-0])rotate([90,180,180])shoulder_arm();
color([1,1,0])translate([-18,-24,-0])rotate([90,0,180])shoulder_arm();


color([1,0.5,0])translate([18+35,-24+2.5,0])rotate([90,45,180])elbow_arm();
color([1,0.5,0])translate([-18-35,-24-2.5,0])rotate([90,135,180])elbow_arm();
