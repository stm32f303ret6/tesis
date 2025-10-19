module shoulder_part(){
l = 45;
h=3.5;
difference(){
hull(){
translate([15,0,0])cylinder(d=8,h=h,$fn=60);
translate([45,0,0])cylinder(d=10,h=h,$fn=60);
}
translate([15,0,-1])cylinder(d=1.6,h=h*2,$fn=30);
translate([28,0,-1])cylinder(d=1.6,h=h*2,$fn=30);
translate([l,0,-1])cylinder(d=3.2,h=h*2,$fn=30);


hull(){
translate([20,0,-1])cylinder(d=3,h=2*h,$fn=60);
translate([23,0,-1])cylinder(d=3,h=2*h,$fn=60);
}
hull(){
translate([33,0,-1])cylinder(d=3,h=2*h,$fn=60);
translate([39,0,-1])cylinder(d=3.6,h=2*h,$fn=60);
}
}
}

shoulder_part();