include <servo.scad>

servo_visible = false;

module servo_screw_mount(){
    difference(){
        base_screw_w = 10;
        translate([0,0,9/2]) cube([base_screw_w, 7, 9], center=true);
        translate([0,0,9/2]) cube([3.1, 10, 10], center=true);
        translate([-10,0,4.8]) rotate([0,90,0]) cylinder(d=3.2,h=20,$fn=60);
    }
}

module single_base_servo_tilt(){
    base_h = 2;
    offset_x = 1;
    offset_y = 1;
    difference(){
        union(){
            if(servo_visible){
                color([0.4,0.4,0.4,0.6])
                    translate([41+offset_x,37+offset_y,10+base_h])
                    rotate([90,0,0])
                    rotate([0,90,0])
                    servo();
            }

            color([0,1,0]) cube([41+offset_x,54+offset_y,base_h]);

            color([1,0,0]) translate([29+offset_x,40+3.5+7+offset_y,base_h]) servo_screw_mount();
            color([1,0,0]) translate([29+offset_x,3.5+offset_y,base_h]) servo_screw_mount();
            translate([15+offset_x, offset_y+51, 0]) cylinder(d=7, h=base_h, $fn=40);
        }

        hull(){
            translate([6+offset_x,10+offset_y,-1]) cylinder(d=5,h=10,$fn=30);
            translate([34+offset_x,10+offset_y,-1]) cylinder(d=5,h=10,$fn=30);
            translate([6+offset_x,40+offset_y,-1]) cylinder(d=5,h=10,$fn=30);
            translate([34+offset_x,40+offset_y,-1]) cylinder(d=5,h=10,$fn=30);
        }

        translate([15+offset_x, offset_y+51, -2]) cylinder(d=4.2, h=40, $fn=40);
    }
}

module base_tilt(){
// Original at (1,1)
single_base_servo_tilt();

// Mirrored to (1,-1)
mirror([0,1,0]) single_base_servo_tilt();

// Mirrored to (-1,1)
mirror([1,0,0]) single_base_servo_tilt();

// Mirrored to (-1,-1)
rotate([0,0,180]) single_base_servo_tilt();

}

//rotate([90,0,0])rotate([0,90,0])servo();
base_tilt();