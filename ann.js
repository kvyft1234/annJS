var ctx = ai_canvas.getContext('2d');

var getOpacityToColor = function(k){
	return "rgb("+ (k+', ').repeat(2) + k +")";
}

var circle = function (x,y,r,opacity) {
	ctx.strokeStyle = getOpacityToColor(opacity);
	ctx.fillStyle = getOpacityToColor(opacity);
	ctx.beginPath();
	ctx.arc(x,y,r, 0, 2 * Math.PI);
	ctx.fill();
}

var line = function(x1,y1,x2,y2,opacity){
	ctx.strokeStyle = getOpacityToColor(opacity);
	ctx.fillStyle = getOpacityToColor(opacity);
	ctx.moveTo(x1, y1);
	ctx.lineTo(x2, y2);
	ctx.stroke();
}

var getPos = function(a,b){
	return [a*50-10, b*50-10];
}

var floorColorRange = function(k){
	if(k<0) return 0;
	else if (k>255) return 255;
	else return Math.floor(k + 0.5);
}

var ai={};
ai.layers = [3,2];
ai.weight = [[[0.1,0.2,0.3], [0.1,0.2,0.3]]];
ai.bias = [[0.1, 0.2]];
ai.value = [[1,2,3], [4,5]];
ai.output = [2,3];

ai.setInput = function (arr) {
	if(this.value[0].length==arr.length) this.value[0]=arr;
	else console.warn("Arguments length doesn't same with input length");
};

ai.setOutput = function (arr) {
	if(this.output.length == arr.length) this.output = arr;
	else console.warn("Arguments length doesn't same with output length");
}

/*ai.createPerceptron = function (layerNumber) {
	if (this.layers.length>layerNumber) {
		this.value[layerNumber].push(0);
		if(layerNumber != 0){
			var k = function () {return 0;};
			this.bias[layerNumber].push(0);
			this.weight[layerNumber].push( Array(this.layers[layerNumber-1]).fill(1).map(k));
		}
		if(layerNumber != this.layers.length - 1){
			this.weight[layerNumber].map(function(){return arguments[0].push(0)},this);
		}
	} else if (this.layers.length == layerNumber){
		this.layers.push(1);
		this.weight.push([]);
		this.weight[layerNumber].push(Array(this.layers[layerNumber - 1]).fill(0));
		this.bias.push([0]);
		this.value.push([0]);
	} else {
		console.log("Unexpected token arguments");
	}
};*/

ai.activationFuntionKind = {
	sigmoid : function (x) {return 1/(1+Math.exp(-x))},
	ReLU : function (x) {return x<0 ? 0:x},
	LeakyReLU : function (x) {return x<0 ? 0.1*x:x},
	tanh : Math.tanh
};

ai.dFunctionDerivative = {
	sigmoid : function (x) {var a = 1/(1+Math.exp(-x)); return a*(1-a);},
	ReLU : function (x) {return x<0 ? 0:1},
	LeakyReLU : function (x) {return x<0 ? 0.1:1},
	tanh : function (x) {return 1/Math.cosh(x)**2}
}

ai.activationFuntion = {
	name : "LeakyReLU",
	execute : ai.activationFuntionKind.LeakyReLU,
	derivative : ai.dFunctionDerivative.LeakyReLU
}

ai.setActivationFunction = function (name) {
	ai.activationFuntion.name = name;
	ai.activationFuntion.execute = eval("ai.activationFuntion."+name);
	ai.activationFuntion.derivative = eval("ai.dFunctionDerivative."+name);
}

ai.setInitialValue = function () {return Math.random()*2-1};

ai.parameterInitialize = function () {
	var a = this.setInitialValue;
	for (var i = 0; i < this.weight.length; i++){
		for (var j = 0; j < this.weight[i].length; j++){
			this.weight[i][j] = this.weight[i][j].map(a);
		}
		this.bias[i] = this.bias[i].map(a);
	}
}

ai.viewResult = function() {
	for (var i = 0; i < this.layers.length; i++){
		for (var j = 0; j < this.layers[i]; j++){
			circle(i*75+50, j*75+35, 25, floorColorRange(127 - this.value[i][j]*100));
		}
	}
	for (var i = 1; i < this.layers.length; i++){
		for (var j = 0; j < this.layers[i]; j++){
			for (var k = 0; k < this.layers[i-1]; k++){
				line((i-1)*75+50, k*75+35, i*75+50, j*75+35, floorColorRange(127 - this.weight[i-1][j][k]*500));
			}
		}
	}
}

ai.cost = function (){
	var sum = 0;
	var lastLayer = this.layers[this.layers.length - 1];
	for (var i = 0; i < lastLayer; i++){
		sum += (this.value[this.layers.length - 1][i]-this.output[i])**2;
	}
	document.querySelectorAll("#cost span")[0].innerText = sum / lastLayer;
	return sum / lastLayer;
};

ai.updataValue = function () {
	var actFunc = this.activationFuntion.execute;
	var sum;
	for (var i = 1; i < this.value.length; i++){
		for (var j = 0; j < this.value[i].length; j++){
			sum = 0;
			for (var k = 0; k < this.value[i-1].length; k++){
				sum += this.weight[i-1][j][k]*this.value[i-1][k];
			}
			this.value[i][j] = actFunc(sum+this.bias[i-1][j]);
		}
	}
	this.viewResult();
	console.log(this.cost());
};

ai.setPerceptron = function (layerArr) {
	var l = layerArr.length;
	this.layers = layerArr;
	this.weight = Array(l - 1).fill(0);
	this.bias = Array(l - 1).fill(0);
	this.value = Array(l).fill(0);
	this.value[0] = Array(layerArr[0]).fill(0);
	this.output = Array(layerArr[l - 1]).fill(0);
	for (var i = 1; i < l; i++){
		this.weight[i - 1] = Array(layerArr[i]).fill(0);
		this.bias[i - 1] = Array(layerArr[i]).fill(0);
		this.value[i] = Array(layerArr[i]).fill(0);
		for (var j = 0; j < layerArr[i]; j++){
			this.weight[i - 1][j] = Array(layerArr[i - 1]).fill(0);
		}
	}
	this.parameterInitialize();
	this.updataValue();
	console.log("set input and output plz\n(as setInput function and setOutput function)");
	this.info();
};

ai.info = function () {return "layer : "+this.layers+[] +"\n"+"activationFuntion : "+ai.activationFuntion.name};

ai.parameterInitialize();
ai.updataValue();
ai.info();
