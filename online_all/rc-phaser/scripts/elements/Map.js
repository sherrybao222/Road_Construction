/*
This is the Map class that generate key map attributes, data saving, 
and update functions the constants location and dis_matrix below are 
for debug purpose when load json is not working
*/

export default class Map{
  //the Phaser Scene will be passed in as a parameter to access time/mouse/location etc
  constructor(mapContent, width, height){

    this.loadMap(mapContent, width, height);
    this.dataInit();
  }

  loadMap(mapContent, width, height){
    
    //this.thisMap     = mapContent[mapID];
    this.cityNr        = mapContent['N'];
    this.radius        = 5; //radius of city
    this.budgetTotal   = mapContent['total'];
    this.budgetRemain  = mapContent['total']; // initialize
    this.x             = mapContent['x'].map(x => x + width/2);
    this.y             = mapContent['y'].map(x => x + height/2);
    this.xy            = this.x.map(function (value, index){return [value, this.y[index]]});
    this.cityStart     = this.xy[0];  
    this.cityDistMat   = mapContent['distance'];
  }

  //------------DATA-STRUCTURE--------------------------------------------------
  dataInit(blockID, trialID, mapID, mouse, time) {

		// basic trial info
		this.blockID       =    blockID;
		this.trialID       =    trialID;
		this.mapID         =    mapID;
    this.condition     =    2; // basic: 2
    // dynamic info
		this.time          =    [time]; 
		this.mousePos      =    [mouse]; 
		this.click         =    [0]; //click indicator
		this.undo          =    [0]; 	

		this.choiceDyn     =    [0]; // start city index
    this.choiceHis     =    [0]; 
    this.choiceLocDyn  =    [this.cityStart]; 
    this.choiceLoc     =    [this.cityStart]; 
    
		this.budgetDyn     =    [this.budgetRemain]; 
    this.budgetHis     =    [this.budgetRemain]; 
    
		this.cityNr        =    [0]; 
		this.check         =    0; //indicator showing if people make valid choice
		this.checkEnd      =    0; 
	  }

  dataChoice(mouse, time) {

		this.time.push(time); 
		this.mousePos.push(mouse); 
		this.click.push(1); //click indicator
		this.undo.push(0); 	
    
    this.choiceDyn.push(this.cityIndex);
    this.choiceHis.push(this.cityIndex);
    this.choiceLocDyn.push(this.cityLoc);
    this.choiceLoc.push(this.cityLoc);

    this.budgetDyn.push(this.budgetRemain);
    this.budgetHis.push(this.budgetRemain);
    
    this.cityNr.push(this.cityNr[this.cityNr.length-1]+1);
    this.check = 0; //change choice indicator after saving them

    delete this.cityIndex;
    delete this.cityLoc;
  }

  dataStatic(mouse, time){

		this.time.push(time); 
		this.mousePos.push(mouse); 
		this.click.push(0); //click indicator
		this.undo.push(0); 	

    this.choiceHis.push(this.choiceDyn[this.choiceDyn.length-1]);
    this.choiceLoc.push(this.choiceLocDyn[this.choiceLocDyn.length-1]);
    this.budgetHis.push(this.budgetDyn[this.budgetDyn.length-1]);

    this.cityNr.push(this.cityNr[this.cityNr.length-1]);
  }

  //---------Check---User---Input-------------------------------------------------
  makeChoice(mouseX,mouseY){
  //do not evaluate the starting point
    for (var i = 1; i < this.xy.length; i++){
      this.mouseDistance = Math.hypot(this.xy[i][0]-mouseX, this.xy[i][1]-mouseY);
      //currently entering input.pointer, not useing the created pointer object
      //this.xy = locations
     
      if (this.mouseDistance <= this.radius && this.choiceDyn.includes(i)==false) {  // cannot choose what has been chosen
        if (this.budgetDyn[this.budgetDyn.length-1] >= 
          this.cityDistMat[i][this.choiceDyn[this.choiceDyn.length-1]]) { // fixed bug of choosing city out of reach in the end

          this.cityIndex = i; //index of chosen city
          this.cityLoc = this.xy[i];//location of chosen city
          this.check = 1; //indicator showing people made a valid choice
          }
      };
    };
  }

 budgetUpdate(){
  //get distance from current choice to previous choice
  let dist = this.cityDistMat[i][this.choiceDyn[this.choiceDyn.length-1]]
  this.budgetRemain = this.budgetDyn[this.budgetDyn.length-1] - dist;
  }

  //check if trial end, not all dots are evaluated right
 checkEnd(){
    // copy distance list for current city
    let cityDistRowCopy = this.cityDistMat[this.choiceDyn[this.choiceDyn.length-1]].slice();
    for (const i of this.choiceDyn) {
      cityDistRowCopy[i] = 0;
    };

    if (cityDistRowCopy.some(i =>
      i < this.budgetDyn[this.budgetDyn.length-1] && i != 0)){
      return true;
    } else {
      return false;
    };
  }

}