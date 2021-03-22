function draw_map(mmap,mouse_x,mouse_y){
    this.budget(mmap,mouse_x,mouse_y);
    this.cities(mmap);
    this.scorebar(mmap);

    //add the two if statments from python
    // if (mmap.choice_dyn.length >= 2){
    //   this.road(mmap);
    // };

    //need to code to update this condition
    if (mmap.check_end_ind){
        this.add.text(100,400,'You are out of budget');
    };
        }
  
function drawCity(scene, mmap, color){
    // define style
    var graphics = scene.add.graphics();
    graphics.fillStyle(color, 0.5);

    // draw cities except for the starting city
    for (var i = 1; i < mmap.xy.length; i++) {
        this.circle.fillCircle(mmap.xy[i][0], mmap.xy[i][1], mmap.radius);
    };

    // draw the starting city
    this.circle.fillCircle(mmap.cityStart[0], mmap.cityStart[1], mmap.radius);
}

function drawBudget(scene, mmap, color, mouseX, mouseY) {
    // define style
    var graphics = scene.add.graphics();
    graphics.lineStyle(4, color, 1.0);

    //budget follow mouse
    let x = mmap.choiceLocDyn[mmap.choiceLocDyn.length - 1][0];
    let y = mmap.choiceLocDyn[mmap.choiceLocDyn.length - 1][1];

    let radians = Math.atan2(mouseY - y, mouseX - x);

    var budgetPosX = x + mmap.budgetDyn[mmap.budgetDyn.length - 1] * Math.cos(radians);
    var budgetPosY = y + mmap.budgetDyn[mmap.budgetDyn.length - 1] * Math.sin(radians);

    //draw budget line
    let line = new Phaser.Geom.Line();
    line.setTo(x, y, budgetPosX, budgetPosY);
    graphics.strokeLineShape(line);
  }

function drawRoad(scene, mmap, color){
    // define style
    var graphics = scene.add.graphics();
    graphics.lineStyle(4, color, 1.0);

    for (var i = 0; i < mmap.choiceLocDyn.length-1; i++) {
        let line = new Phaser.Geom.Line(
        mmap.choiceLocDyn[i][0],mmap.choiceLocDyn[i][1],
        mmap.choiceLocDyn[i+1][0],mmap.choiceLocDyn[i+1][1]);
        graphics.strokeLineShape(line);
    };
}

class scorebar {

    constructor(mmap){
        //score bar parameters
        this.width = 100;
        this.height = 400;
        this.box = 12;
        this.top = 50; //distance to screen top
    
        this.box_center(); //center for labels
        this.incentive(); //calculate incentive: N^2
        this.indicator(mmap); //incentive score indicator, merged with older arrow function
        this.number();
        }
    
    box_center(){
        this.box_height = this.height / this.box
        this.center_list = []
        this.uni_height = this.box_height / 2
        this.x = this.width / 2 + 600  //larger the number, further to right, 1300

        for (var i=0; i<this.box; i++){
            let loc = [this.x, i * this.box_height + this.uni_height];
            this.center_list.push(loc);
        };
    }
  
    incentive(){
        this.score = Array.from(Array(this.box).keys());
        this.incentive_score = [];
        for (let i of this.score){
            i = (i**2) * 0.01;
            this.incentive_score.push(i);
        };
        };
  
    indicator(mmap){
        this.indicator_loc = this.center_list[mmap.n_city[mmap.n_city.length-1]];
        this.indicator_loc_best = this.center_list[Math.max(mmap.n_city)];

        //create triangle arrow and define style
        this.triangle = this.add.graphics();
        this.triangle.fillStyle(grey);

        //arrow parameter
        let point = [this.indicator_loc[0] - 30, this.indicator_loc[1]+this.top+10];
        let v2 = [point[0] - 10, point[1] + 10];
        let v3 = [point[0] - 10, point[1] - 10];
        this.triangle.fillTriangle(point[0], point[1], v2[0],v2[1],v3[0],v3[1]);
        }
  
    //rendering score Bar
    number(){
        //create rectangle and define style
        this.rect = this.add.graphics();

        let left = this.center_list[0][0] - 25;
        let hex_c = [0x66CC66,0x74C366,0x82B966,0x90B066,
                0x9EA766,0xAC9E66,0xB99466,0xC78B66,
                0xD58266,0xE37966,0xF16F66,0xFF6666] //color list

        for (var i=0; i<this.box; i++){
        let loc = this.center_list[i];
        let text = this.incentive_score[i];
        //score bar outline
        this.rect.fillStyle(grey);
        this.rect.fillRect(left,loc[1]+this.top-this.uni_height,
        this.width,this.box_height);
        //score bar fill
        this.rect.fillStyle(hex_c[i]);
        this.rect.fillRect(left,loc[1]+this.top-this.uni_height+2,
        this.width,this.box_height);

        this.add.text(loc[0],loc[1]+this.top,text);
        };

        // scorebar title
        this.add.text(this.center_list[0][0]-20,this.center_list[0][1]+this.top-50,
        'Bonus in dollars');
    }
}
  