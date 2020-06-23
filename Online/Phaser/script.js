// set up the configuration of the window/game, also the parameter

var config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    scene: [MainTask],
        // {
        //     // preload: preload,
        //     // create: create,
        //     // update: update,
        // }
    }

//this set up the canvas and game framework
var game = new Phaser.Game(config);

//scenes are used to organize content


function preload ()
{
    this.load.setBaseURL('http://labs.phaser.io');
    this.load.image('sky', 'assets/skies/space3.png');
    this.load.image('logo', 'assets/sprites/phaser3-logo.png');
    this.load.image('red', 'assets/particles/red.png');
}

//this create function is like Draw in pygame
function create ()
{
    // this.scene.start('MainTask');
    //define your graphics/style
    // this.graphics = this.add.graphics();
    // this.graphics.lineStyle(5, 0xFF00FF, 1.0);
    // this.graphics.moveTo(100, 100);
    // this.graphics.lineTo(200, 200);
    // //create filled solide style
    // this.graphics.fillStyle(0xFF00FF,.5);
    // // this.graphics.fillCircle(0,40,6);
    //
    // //this create line object, and then use stroke to draw
    // // different from graphics somehow (start,end)
    // var line2 = new Phaser.Geom.Line(20,40,50,80);
    // console.log(line2.getPointA()); //return the coordinate
    //
    // var locations = [[10,20],[49,80],[22,59],[34,60]];
    // // console.log(locations[0]);
    //
    // for (var i=0; i<locations.length; i++){
    //       var x = locations[i][0];
    //       var y = locations[i][1];
    //       this.circle = this.add.graphics();
    //       this.circle.fillCircle(x,y,6);
    //       // this.graphics.strokeCircle(x,y,6);
    //     }
    //
    // //activate your graphics, stroke = draw
    // this.graphics.strokePath();
    // this.graphics.strokeCircle(100,100,6);
    // this.graphics.strokeLineShape(line2);
    //
    // this.add.text(20,20,"Hello, World");
    //
    // //mouse position
    // var input = this.input;
    // var mouse = this.input.mousePointer;
    // console.log(mouse.x);

}

// function update(){
//   //angle between line and mouse
//   console.log(mouse.x, mouse.y);
//   // let angle = Phaser.Math.Angle.Between(line.x,line.y,mouse.x,mouse.y);
//   // line.setRotation(angle+Math.PI/2);
// }
