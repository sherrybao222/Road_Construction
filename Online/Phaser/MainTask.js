//this create context/scene seperatly to be called in the main file

//color constants
const grey = 0xFAF7F6;
const green = 0xA2EF4C;

//city/response list for debug
const locations = [[10,20],[49,80],[22,59],[300,300]];
choice_locdyn = [[10,20],[49,80],[22,59],[300,300]];
budget_dyn = [60,40,100];

class MainTask extends Phaser.Scene {
    constructor() {
        super('MainTask');
    }
    preload()
    {

    }
    create(){
        console.log("Ready!");
        //call your major functions
        // this.style();
        this.cities();
        // this.road();
        this.budget();

        //initialite new mouse input
        this.input = new Phaser.Input.InputManager(game, config);
        this.mouse = new Phaser.Input.Mouse.MouseManager(this.input);
    };

    style(){
      // step up general graphics collections
      this.graphics = this.add.graphics();
      //line style
      // this.graphics.lineStyle(3, green,1.0);
      //circle style
      // this.graphics.fillStyle(grey,.5);
    };

    //so far these are simple function, no agruament yet
    cities(){
        //create visuals and define style
        this.city = this.add.graphics();
        this.city.fillStyle(grey,.5);

        for (var i=1; i<locations.length; i++){
          this.x = locations[i][0];
          this.y = locations[i][1];
          this.city.fillCircle(this.x,this.y,6);
        };
        this.city.fillCircle(locations[0][0],locations[0][1],6);
    };

    road(){
      //initialite new mouse input
      // this.input = new Phaser.Input.InputManager(game, config);
      // this.mouse = new Phaser.Input.Mouse.MouseManager(this.input);
      // console.log(game.input.mousePointer.x);

      this.graphics.moveTo(100, 100);
      this.graphics.lineTo(200, 200);

      //this create line object, and then use stroke to draw
      // different from graphics somehow (start,end)
      // this.line2 = new Phaser.Geom.Line(20,40,this.x,this.y);
      // console.log(line2.getPointA()); //return the coordinate

      //activate your graphics, stroke = draw
      this.graphics.strokePath();
      this.graphics.fillCircle(100,100,6);
      // this.graphics.strokeLineShape(line2);
    };

    budget(){
      //create visuals and define style
      this.budget_line = this.add.graphics();
      this.budget_line.lineStyle(4, green, 1.0);

      //mouse input setup
      this.mouse_x = game.input.mousePointer.x;
      this.mouse_y = game.input.mousePointer.y;

      //current city loc: mmap.choice_locdyn[-1][0]
      //JS negative index is different
      let x = choice_locdyn[choice_locdyn.length - 1][0];
      let y = choice_locdyn[choice_locdyn.length - 1][1];

      //budget follow mouse
      let cx = this.mouse_x - x;
      let cy = this.mouse_y - y;
      let radians = Math.atan2(cy,cx);

      //mmap.budget_dyn[-1]
      this.budget_pos_x = x + budget_dyn[budget_dyn.length - 1] * Math.cos(radians);
      this.budget_pos_y = y + budget_dyn[budget_dyn.length - 1] * Math.sin(radians);

      //draw budget line
      let line = new Phaser.Geom.Line(x,y,this.budget_pos_x,this.budget_pos_y);
      this.budget_line.strokeLineShape(line);
    };

    update(){
      this.add.text(20,20,"Hello, World");

      //destroy as a function to update per frame 
      this.budget_line.destroy();
      this.budget();
    };
}
