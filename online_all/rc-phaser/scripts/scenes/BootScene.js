import {validCode} from '../configs/validCode.js';
import {basicTrainMap} from '../configs/basicTrainMap.js';
import {basicMap} from '../configs/basicMap.js';
import {undoTrainMap} from '../configs/undoTrainMap.js';
import {undoMap} from '../configs/undoMap.js';


export default class BootScene extends Phaser.Scene {

	constructor() {
		super('BootScene');
	}

	preload() {
		// get exp start time
		var time = new Date();
		this.registry.set('expStartTime', formatDate(time));
		
		// add config
		this.registry.set('timeLimit', 10000) // ms

		// add configurations to registry
		this.registry.set('basicTrainMap', basicTrainMap);
		this.registry.set('basicMap', basicMap);
		this.registry.set('undoTrainMap', undoTrainMap);
		this.registry.set('undoMap', undoMap);
		this.registry.set('validCode', validCode);
		
		this.registry.set('gameTrialNr',  4);
		this.registry.set('trainTrialNr', 2);
		this.registry.set('basicNr', 0); // initialize
		this.registry.set('undoNr', 0); // initialize
		this.registry.set('trialCounter', 0) // intialize how many trials have done

		// create random index for group and single condition trials
		var basicInd  = Array.from(Array(this.registry.values.gameTrialNr/2).keys());
		this.shuffleArray(basicInd)
		//localStorage.setItem('groupInd', JSON.stringify(groupInd));
		this.registry.set('basicInd', basicInd); 

		var undoInd = Array.from(Array(this.registry.values.gameTrialNr/2).keys());
		this.shuffleArray(undoInd)
		// localStorage.setItem('singleInd',  JSON.stringify(singleInd));
		this.registry.set('undoInd', undoInd); 
		
		var cond = [2,3,3,2]
		//localStorage.setItem('oneAll', JSON.stringify(oneAll));
		this.registry.set('cond', cond); 
		
	}

	create() {		
		this.scene.start("TitleScene");
	}

	// helper functions
	shuffleArray(array) {
		for (let i = array.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[array[i], array[j]] = [array[j], array[i]];
		}
	}

	shuffleBlock(NrBlock) {
		var orderBlock = new Array(NrBlock).fill(0);
		var hyperFlip;
		     

		for (var i = 0; i < NrBlock/2; i++) {
			hyperFlip = Math.round(Math.random());
			if (hyperFlip === 0){
				orderBlock[i*2] = 1; //group
				orderBlock[i*2+1] = 2; //single
			}else{
				orderBlock[i*2] = 2; 
				orderBlock[i*2+1] = 1; 
			}
		}

		return orderBlock
	}
}