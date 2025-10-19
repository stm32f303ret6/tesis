include <../two_servo_base.scad>
include <../servo.scad>
include <../shoulder_arm.scad>
include <../elbow_arm.scad>

module mod(side){
translate([0,20,0]){
rotate([0,90,270]){
color([0.2,0.2,0.2])translate([10,11.05,15])servo();
color([0.2,0.2,0.2])translate([10,-11.05,15])servo();
color([0.2,0.2,0.2])translate([0,-36,-2])rotate([90,-90,180])servo();
color([0.0,0.9,0])translate([0,0,-2])two_servo_base();
}

color([0,0.9,0])translate([11,-24,-10])rotate([90,180,180])shoulder_arm();
color([0,0.9,0])translate([-11,-24,-10])rotate([90,0,180])shoulder_arm();

color([0,0.9,0])translate([11+40,-24+2.5,-10])rotate([90,30,180])elbow_arm();
color([0,0.9,0])translate([-11-40,-24-2.5,-10])rotate([90,150,180])elbow_arm();
}
}

translate([0,0,0])rotate([0,0,0])mod(1);
translate([-155,,0])rotate([0,0,0])mod(2);