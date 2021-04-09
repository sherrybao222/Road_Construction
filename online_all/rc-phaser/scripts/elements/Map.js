/*
This is the Map class that generate key map attributes, data saving, 
and update functions the constants location and dis_matrix below are 
for debug purpose when load json is not working
*/

class Map{
  //the Phaser Scene will be passed in as a parameter to access time/mouse/location etc
  constructor(mapContent, width, height, blockID, trialID, mapID, mouse, time){
    this.loadMap(mapContent, width, height);
    this.dataInit(blockID, trialID, mapID, mouse, time);
  }

  loadMap(mapContent, width, height){
    
    this.cityNr        = mapContent['N'];
    this.radius        = 5; //radius of city
    this.budgetTotal   = mapContent['total'];
    this.budgetRemain  = mapContent['total']; // initialize
    this.x             = mapContent['x'].map(x => x + width/2);
    this.y             = mapContent['y'].map(y => y + height/2);
    var y              = this.y;
    this.xy            = this.x.map(function (value, index){
                                              return [value, y[index]]});
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
}

function dataChoice(mmap, mouse, time) {
  
  mmap.time.push(time); 
  mmap.mousePos.push(mouse); 
  mmap.click.push(1); //click indicator
  mmap.undo.push(0); 	
  
  mmap.choiceDyn.push(mmap.cityIndex);
  mmap.choiceHis.push(mmap.cityIndex);
  mmap.choiceLocDyn.push(mmap.cityLoc);
  mmap.choiceLoc.push(mmap.cityLoc);

  mmap.budgetDyn.push(mmap.budgetRemain);
  mmap.budgetHis.push(mmap.budgetRemain);
  
  mmap.cityNr.push(mmap.cityNr[mmap.cityNr.length-1]+1);
  mmap.check = 0; //change choice indicator after saving them

  delete mmap.cityIndex;
  delete mmap.cityLoc;
}

function dataStatic(mmap, mouse, time){

  mmap.time.push(time); 
  mmap.mousePos.push(mouse); 
  mmap.click.push(0); //click indicator
  mmap.undo.push(0); 	

  mmap.choiceHis.push(mmap.choiceDyn[mmap.choiceDyn.length-1]);
  mmap.choiceLoc.push(mmap.choiceLocDyn[mmap.choiceLocDyn.length-1]);
  mmap.budgetHis.push(mmap.budgetDyn[mmap.budgetDyn.length-1]);

  mmap.cityNr.push(mmap.cityNr[mmap.cityNr.length-1]);
}

  //---------Check---User---Input-------------------------------------------------
function makeChoice(mmap, mouseX, mouseY){
//do not evaluate the starting point
  for (var i = 1; i < mmap.xy.length; i++){
    mmap.mouseDistance = Math.hypot(mmap.xy[i][0]-mouseX, mmap.xy[i][1]-mouseY);
    //console.log(this.mouseDistance)      
    if (mmap.mouseDistance <= mmap.radius && mmap.choiceDyn.includes(i)==false) {  // cannot choose what has been chosen
      if (mmap.budgetDyn[mmap.budgetDyn.length-1] >= 
        mmap.cityDistMat[i][mmap.choiceDyn[mmap.choiceDyn.length-1]]) { // fixed bug of choosing city out of reach in the end

        mmap.cityIndex = i; //index of chosen city
        mmap.cityLoc = mmap.xy[i];//location of chosen city
        mmap.check = 1; //indicator showing people made a valid choice
        }
    };
  };
}

function budgetUpdate(mmap){
  //get distance from current choice to previous choice
  let dist = mmap.cityDistMat[mmap.cityIndex][mmap.choiceDyn[mmap.choiceDyn.length-1]]
  mmap.budgetRemain = mmap.budgetDyn[mmap.budgetDyn.length-1] - dist;
}

function checkEnd(mmap) {
  // copy distance list for current city
  let cityDistRowCopy = mmap.cityDistMat[mmap.choiceDyn[mmap.choiceDyn.length-1]].slice();
  for (var i in mmap.choiceDyn) {
    cityDistRowCopy[i] = 0;
  };

  if (cityDistRowCopy.some(i =>
    i < mmap.budgetDyn[mmap.budgetDyn.length-1] && i != 0)){
    return true; // not end
  } else {
    return false; // end
  };
}

function getTime() {
  //make a new date object
  let d = new Date();
  //return the number of milliseconds since 1 January 1970 00:00:00.
  return d.getTime();
}
