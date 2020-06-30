//this create context/scene seperatly to be called in the main file

//color constants
const grey = 0xFAF7F6;
const green = 0xA2EF4C;
const width = 1000;
const  height = 900

//city/response list for debug
const locations = [[200,150],[270,90],[460,200],[300,300]];
let choice_locdyn = [[200,150]];
let choice_dyn = [];
let budget_dyn = [60,40,100];

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
        // this.road();
        this.budget();
        this.cities();
        this.load_map();

        //create pointer
        // this.pointer = new Phaser.Input.Pointer();
        this.pointer = this.input.activePointer;
        //generate circle map parameters
        this.N = 20; //totall city number, including start
        this.radius = 10; //radius of city
        this.total = 400; //total budget
        this.budget_remain = 400; //remaining budget_dyn

        this.R = 400*400; //circle radius' sqaure

    };

    data_rc(){
      // this.choice_dyn = [];
      choice_locdyn.push(this.city);
      choice_dyn.push(this.city);
      this.check = 0; //change choice indicator after saving them
      console.log(choice_dyn);
    };

    //load json map into javascript
    load_map(map_content, map_id){
        //this.loadmap = map_content[map_id];
        this.order = NaN;

        //this.N = this.loadmap['N'];
        this.radius = 5;
        //this.total = this.loadmap['total'];
        //this.budget_remain = this.loadmap['total'];

        //this.R = this.loadmap['R'];
        //this.r = this.loadmap['r'];
        //this.phi = this.loadmap['phi'];
        // this.x = this.load_map['x'].map(x => x[0] + width/2);
        // this.y = this.load_map['y'].map(x => x[1] + height/2);
        // this.xy = this.x.map(x => [x,this.y[this.x.indexOf(x)]]);
        // console.log(this.xy); //not working join list is working

    };

    //so far these are simple function, no agruament yet
    cities(){
        //create visuals and define style
        this.circle = this.add.graphics();
        this.circle.fillStyle(grey,.5);

        let city_list = [];
        for (var i=1; i<locations.length; i++){
          this.x = locations[i][0];
          this.y = locations[i][1];
          let city = this.circle.fillCircle(this.x,this.y,6);
          city_list.push(city);
        };
        let start = this.circle.fillCircle(locations[0][0],locations[0][1],6);
        city_list.push(start);
        // this.city.setInteractive(city_list,this.onObjectClicked);
        this.circle.fillCircle(30,50,6);
        // this.city.setInteractive(this.city,this.onObjectClicked);
        // this.input.on('gameobjectdown',this.onObjectClicked);
    };

    // onObjectClicked(pointer,city_list){
    //   // if (pointer == city_list.prototype.some()){
    //     // console.log("clicked");
    //   // };
    // };

    road(){
      //function name and this.name don't use the same
      //otherwise lead to naming bug
      //create road and define style
      this.line = this.add.graphics();
      this.line.lineStyle(4, grey, 1.0);

      //click and draw
      // this.input.on('pointerdown', function(pointer){
      //     // console.log('down');
      //     let line2 = new Phaser.Geom.Line(
      //       choice_locdyn[choice_locdyn.length-1][0],
      //       choice_locdyn[choice_locdyn.length-1][1],
      //       pointer.x,pointer.y);
      //     this.road.strokeLineShape(line2);
      //     // this.add.text(pointer.x,pointer.y,"Road Construction")
      //     },this);

      // draw the connected cities from "mmap.choice_locdyn"
      for (var i=0; i<choice_locdyn.length-1; i++){
          let line = new Phaser.Geom.Line(
          choice_locdyn[i][0],choice_locdyn[i][1],
          choice_locdyn[i+1][0],choice_locdyn[i+1][1]);
          this.line.strokeLineShape(line);
      };
    };

    budget(){
      //create budget line and define style
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

    make_choice(pointer){
      // let x2 = this.mouse_x;
      // let y2 = this.mouse_y;
      for (var i=1; i<locations.length; i++){
        this.mouse_distance = Math.hypot(locations[i][0]-this.pointer.x,
          locations[i][1]-this.pointer.y);
        // 3 is based on visual testing, not based on radius
        if (this.mouse_distance <= 3 && choice_dyn.includes(choice_dyn[i])==false){
          // console.log(choice_dyn.includes(i));
          // console.log(this.mouse_distance);
          this.index = i; //index of chosen city
          this.city = locations[i];
          // this.city = this.xy[i]; //location of chosen city
          this.check = 1; //indicator showing people made a valid choice
        };
      };
    };

    update(){
      this.add.text(20,20,"Road Construction");
      //destroy as a function to update per frame
      this.budget_line.destroy();
      this.budget();
      // this.make_choice();

      this.input.on('pointerdown', function (pointer){
        if (this.pointer.leftButtonDown()){
            this.make_choice();
            if (this.check == 1){
                // console.log("left down");
                this.data_rc();
                this.road();
            };
        };
      }, this);
      // console.log(choice_dyn);
      // this.make_choice(); //this enable live update while playing

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

    };
}
