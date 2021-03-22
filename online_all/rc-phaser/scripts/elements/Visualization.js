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

    constructor(scene,mmap,color){
        //score bar parameters
        this.barWidth = 100;
        this.barHeight = 400;
        this.nrBox = 12;
        this.distToTop = 50; //distance to screen top
    
        this.boxCenter(scene); //center for labels
        this.incentive(); //calculate incentive: N^2
        this.drawScorebar(scene);
        this.indicator(scene,mmap,color);
        }
    
    boxCenter(scene){
        this.boxHeight = this.barHeight / this.nrBox;
        this.centerList = [];
        this.halfHeight = this.boxHeight / 2;
        this.x = this.barWidth / 2 + scene.cameras.main.width / 2;  //larger the number, further to right

        for (var i = 0; i < this.nrBox; i++){
            let boxLoc = [this.x, i * this.boxHeight + this.halfHeight];
            this.centerList.push(boxLoc);
        };
    }
  
    incentive(){
        this.Score = [];
        for (var i = 0; i < this.nrBox; i++){
            i = (i**2) * 0.01;
            this.Score.push(i);
        };
    };

    //rendering score Bar
    drawScorebar(scene){
        //create rectangle and define style
        this.rect = scene.add.graphics();

        let barLeft = this.centerList[0][0] - 25;
        let colorBox = [0x66CC66,0x74C366,0x82B966,0x90B066,
                    0x9EA766,0xAC9E66,0xB99466,0xC78B66,
                    0xD58266,0xE37966,0xF16F66,0xFF6666] //color list

        for (var i = 0; i < this.nrBox; i++){

            let boxLoc = this.centerList[i];
            let text = this.Score[i];

            //score bar outline
            scene.rect.fillStyle(grey);
            scene.rect.fillRect(barLeft, boxLoc[1] + this.distToTop - this.halfHeight, this.barWidth, this.boxHeight);

            //score bar fill
            scene.rect.fillStyle(colorBox[i]);
            scene.rect.fillRect(barLeft, boxLoc[1] + this.distToTop - this.halfHeight + 2, this.barWidth, this.boxHeight); //? why + 2

            scene.add.text(boxLoc[0], boxLoc[1] + this.distToTop, text);

        };

        // scorebar title
        scene.add.text(this.centerList[0][0]-20, this.centerList[0][1] + this.distToTop - 50,
                      'Bonus in dollars');
    }

    indicator(scene,mmap,color){
        this.indicatorLoc = this.centerList[mmap.cityNr[mmap.cityNr.length-1]];
        this.indicatorLocBest = this.centerList[Math.max(mmap.cityNr)]; // undo 

        //create triangle arrow and define style
        this.triangle = scene.add.graphics();
        this.triangle.fillStyle(color);

        //arrow parameter
        let point = [this.indicatorLoc[0] - 30, this.indicatorLoc[1] + this.distToTop + 10];
        let v2 = [point[0] - 10, point[1] + 10];
        let v3 = [point[0] - 10, point[1] - 10];
        scene.triangle.fillTriangle(point[0], point[1], v2[0], v2[1], v3[0], v3[1]);
        }
  

}
  