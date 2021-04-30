// basic condition training session
export default class GameScene extends Phaser.Scene {

  constructor() {
      super('GameScene');
  }

  init() {
    // time 
		this.start = getTime();

    // set colors
    this.black = 0x000000;
    this.green = 0x25A35A;
    this.red   = 0xB93D30;
		this.colorText     = '#1C2833'; // black
		this.warnColorText = '#943126'; // red

		// trial parameters
		this.basicNr   = parseInt(this.registry.values.basicNr); // localStorage.getItem('groupNr')
		this.undoNr  = parseInt(this.registry.values.undoNr); // localStorage.getItem('singleNr')
		
		if ((this.basicNr+this.undoNr) != this.registry.values.trialCounter) { // in case there is mismatch
			this.registry.values.trialCounter--;
		}

		this.trialCounter  = this.registry.values.trialCounter;
		this.blockInd      = Math.floor(this.trialCounter/10);
		this.cond        = this.registry.values.cond[this.blockInd]; // JSON.parse(localStorage.getItem('oneAll'))[this.blockInd]	

		this.basicInd         = this.registry.values.basicInd;//JSON.parse(localStorage.getItem('groupInd'));
		this.undoInd        = this.registry.values.undoInd;//JSON.parse(localStorage.getItem('singleInd'));

		if (this.cond === 2){
			this.trialInd     = this.basicInd[this.basicNr];
      this.mapContent = this.registry.values.basicTrainMap[this.trialInd];    
		} else {
			this.trialInd     = this.undoInd[this.undoNr];
      this.mapContent = this.registry.values.undoTrainMap[this.trialInd];    
		}

		this.undoObj = this.input.keyboard.addKey('z');  // Get key object
  }

  preload(){
  }
    
  create(){
    // time + mouse
    var time = new Date();
    var elapsed = time.getTime()-this.start; 
    var mouse = [this.input.mousePointer.x, this.input.mousePointer.y]
    // create map and data saving structure
    this.mapInfo = new Map(this.cond, this.mapContent, this.cameras.main.width, this.cameras.main.height, 1, 1, 1, mouse, elapsed);     //mapContent, width, height, blockID, trialID, mapID, mouse, time    

    // draw cities
    drawCity(this, this.mapInfo, this.black, this.red);

    // save static data every 1s
    this.time.addEvent({
      delay: 1000,                // ms
      callback: this.staticDataTimer,      // save static data
      callbackScope: this,
      loop: true
    });

    // draw scorebar
    this.scorebar = new scorebar(this, this.mapInfo, this.black)

    // draw budget and move
    this.input.on('pointermove', function (pointer) {

      if (typeof this.line !== 'undefined') {
        this.budgetGraphics.clear()
      } else {
        this.line = new Phaser.Geom.Line();
      } 

      // define style
      this.budgetGraphics = this.add.graphics();
      this.budgetGraphics.lineStyle(4, this.green, 1.0);

      //budget follow mouse
      let x = this.mapInfo.choiceLocDyn[this.mapInfo.choiceLocDyn.length - 1][0];
      let y = this.mapInfo.choiceLocDyn[this.mapInfo.choiceLocDyn.length - 1][1];

      let radians = Math.atan2(pointer.y - y, pointer.x - x);

      var budgetPosX = x + this.mapInfo.budgetDyn[this.mapInfo.budgetDyn.length - 1] * Math.cos(radians);
      var budgetPosY = y + this.mapInfo.budgetDyn[this.mapInfo.budgetDyn.length - 1] * Math.sin(radians);
      
      //draw budget line
      this.line.setTo(x, y, budgetPosX, budgetPosY);
      this.budgetGraphics.strokeLineShape(this.line);

    },this);   

    // make choice
    this.input.on('pointerdown', function (pointer){

      var time = new Date();
      var elapsed = time.getTime()-this.start; 
  
      if (pointer.leftButtonDown()){
        var notEnd = checkEnd(this.mapInfo);
        if (notEnd){ // if the trial not end
          makeChoice(this.mapInfo, pointer.x, pointer.y);

          if (this.mapInfo.check == 1){ // if this is valid choice
            budgetUpdate(this.mapInfo);
            dataChoice(this.mapInfo,[pointer.x,pointer.y],elapsed); // time
            drawRoad(this, this.mapInfo, this.black)

            this.scorebar.triangle.clear();
            this.scorebar.indicator(this,this.mapInfo,this.black);
          } else {
            dataStatic(this.mapInfo, [pointer.x,pointer.y], elapsed); // time
          };
        } else {
          this.add.text(20, 100, "You are out of budget!", { fontFamily: 'Comic Sans MS', fontSize: '26px', color: this.warnColorText});
        };
      };
    }, this);

    // add text
    this.add.text(20, 50, "Press RETURN to submit", { fontFamily: 'Comic Sans MS', fontSize: '26px', color: this.colorText});


  }

	update() {	   
		// Is key down?
		if (this.cond === 3 && this.undoObj.isDown && this.mapInfo.choiceDyn.length > 1) {
      var time = new Date();
      var elapsed = time.getTime()-this.start; 
      var mouse = [this.input.mousePointer.x, this.input.mousePointer.y]
  
      this.scorebar.triangle.clear();
      this.scorebar.indicator(this,this.mapInfo,this.black);
      dataUndo(this.mapInfo, mouse, elapsed); // time
    }
	}	

  staticDataTimer(){
    var time = new Date();
    var elapsed = time.getTime()-this.start; 
    var mouse = [this.input.mousePointer.x, this.input.mousePointer.y]
    dataStatic(this.mapInfo, mouse, elapsed)

  };
}

