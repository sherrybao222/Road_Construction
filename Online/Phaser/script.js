// set up the configuration of the window/game, also the parameter

var config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    scene: [Instruction,MainTask]
        // {
        //     // preload: preload,
        //     // create: create,
        //     // update: update,
        // }
    }

//this set up the canvas and game framework
var game = new Phaser.Game(config);

//scenes are used to organize content

//
// function preload ()
// {
//     this.load.setBaseURL('http://labs.phaser.io');
//     this.load.image('sky', 'assets/skies/space3.png');
//     this.load.image('logo', 'assets/sprites/phaser3-logo.png');
//     this.load.image('red', 'assets/particles/red.png');
// }
//
// //this create function is like Draw in pygame
// function create ()
// {
//
// }

// function update(){
//   //angle between line and mouse
//   console.log(mouse.x, mouse.y);
//   // let angle = Phaser.Math.Angle.Between(line.x,line.y,mouse.x,mouse.y);
//   // line.setRotation(angle+Math.PI/2);
// }


//check choice
// if (this.pointer.noButtonDown() == false){
//   console.log("clicked");
// };
// this.input.on('pointerdown', function () {
//   // console.log(this.check);
//   this.make_choice(); //once made on valid choice, it's always 1
//   if (this.check == 1){
//     console.log('down');
//     // this.data();
//   //   this.add.text(pointer.x,pointer.y,"Road Construction");
//   };
// },this);
