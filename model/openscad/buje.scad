module buje(){
h=2;
difference(){
cylinder(d=4.2,h=2,$fn=20);
translate([0,0,-1])cylinder(d=3.2,h=4,$fn=60);
}  
}
buje();