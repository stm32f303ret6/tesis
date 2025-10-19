/**
 *  Parametric servo arm generator for OpenScad
 *  Générateur de palonnier de servo pour OpenScad
 *
 *  Copyright (c) 2012 Charles Rincheval.  All rights reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  Last update :
 *  https://github.com/hugokernel/OpenSCAD_ServoArms
 *
 *  http://www.digitalspirit.org/
 */

arm_length = 30;

arm_count = 1; // [1,2,3,4,5,6,7,8]

//  Clear between arm head and servo head (PLA: 0.3, ABS 0.2)
SERVO_HEAD_CLEAR = 0.2; // [0.2,0.3,0.4,0.5]

$fn = 40 / 1;

/**
 *  Head / Tooth parameters
 *  Futaba 3F Standard Spline
 *  http://www.servocity.com/html/futaba_servo_splines.html
 *
 *  First array (head related) :
 *  0. Head external diameter
 *  1. Head heigth
 *  2. Head thickness
 *  3. Head screw diameter
 *
 *  Second array (tooth related) :
 *  0. Tooth count
 *  1. Tooth height
 *  2. Tooth length
 *  3. Tooth width
 */
FUTABA_3F_SPLINE = [
    [5.92, 4, 1.1, 2.5],
    [25, 0.3, 0.7, 0.1]
];

module servo_futaba_3f(length, count) {
    servo_arm(FUTABA_3F_SPLINE, [length, count]);
}

/**
 *  If you want to support a new servo, juste add a new spline definition array
 *  and a module named like servo_XXX_YYY where XXX is servo brand and YYY is the
 *  connection type (3f) or the servo type (s3003)
 */

module servo_standard(length, count) {
    servo_futaba_3f(length, count);
}

/**
 *  Tooth
 *
 *    |<-w->|
 *    |_____|___
 *    /     \  ^h
 *  _/       \_v
 *   |<--l-->|
 *
 *  - tooth length (l)
 *  - tooth width (w)
 *  - tooth height (h)
 *  - height
 *
 */
module servo_head_tooth(length, width, height, head_height) {
    linear_extrude(height = head_height) {
        polygon([[-length / 2, 0], [-width / 2, height], [width / 2, height], [length / 2,0]]);
    }
}

/**
 *  Servo head
 */
module servo_head(params, clear = SERVO_HEAD_CLEAR) {

    head = params[0];
    tooth = params[1];

    head_diameter = head[0];
    head_heigth = head[1];

    tooth_count = tooth[0];
    tooth_height = tooth[1];
    tooth_length = tooth[2];
    tooth_width = tooth[3];


    cylinder(r = head_diameter / 2 - tooth_height + 0.03 + clear, h = head_heigth);

    for (i = [0 : tooth_count]) {
        rotate([0, 0, i * (360 / tooth_count)]) {
            translate([0, head_diameter / 2 - tooth_height + clear, 0]) {
                servo_head_tooth(tooth_length, tooth_width, tooth_height, head_heigth);
            }
        }
    }
}

/**
 *  Servo hold
 *  - Head / Tooth parameters
 *  - Arms params (length and count)
 */
module servo_arm(params, arms) {

    head = params[0];
    tooth = params[1];

    head_diameter = head[0];
    head_heigth = head[1];
    head_thickness = head[2];
    head_screw_diameter = head[3];

    tooth_length = tooth[2];
    tooth_width = tooth[3];

    arm_length = arms[0];
    arm_count = arms[1];


servo_head(params);
}

module servo_toothed_head() {
  translate([0,0,0])
        servo_standard(arm_length, arm_count);
}

//servo_toothed_head();
