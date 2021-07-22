
var mkRange = function(a){let sum=0; return Array(a).fill(1).map(function(k){return sum++;})};
Math.rand = function(){let r=Math.random(); return Math.log(r/(1-r))};
JSON.new = function(a){return JSON.parse(JSON.stringify(a));};
Array.sum = function(a){let sum=0; Array.num(a.length).map(function(k){sum += a[k]}); return sum;};
var mkTensor = function(a,...b){return b.length==0 ? Array(a).fill(0):Array(a).fill(0).map(i=>mkTensor(...b))};
var mkArray2 = function(arr){let r=mkTensor(arr.length); for(let i=0; i<arr.length; i++){r[i]=mkTensor(arr[i])} return r;};
var fillRandom = function(arr){return arr.map(i => Math.random()).map(i => Math.log(i/(1-i)))};
var cl = function (a, ...b){let c = Array.from(arguments).map(i=>JSON.new(i)); console.log.apply(this,c); return c[0];};

const AI = {};

AI.activationFunction = {};
AI.ANN = {};

AI.activationFunction.leakyReLU = function (x){return x>0 ? x:x*0.1};
AI.activationFunction.sigmoid = function (x){return 1/(1+Math.E**(-x))};
AI.activationFunction.tanh = Math.tanh;
AI.activationFunction.ReLU = x => x>0 ? x:0;

AI.ANN.variable = {};
AI.ANN.variable.layer = [];
AI.ANN.variable.perceptron = [];
AI.ANN.variable.weight = [];
AI.ANN.variable.bias = [];
AI.ANN.variable.z = [];
AI.ANN.variable.activationFunction = [];
AI.ANN.setLayer = function (layer){
	this.variable.layer = layer;
	let l = layer;
	let ll = l.length;
	let v = this.variable;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	let af = AI.activationFunction;
	let lr = af.leakyReLU;
	let sm = af.sigmoid;
	p = mkArray2(l);
	w = mkRange(ll-1).map(i => mkTensor(l[i+1]).map(j => fillRandom(mkTensor(l[i]))));
	b = mkRange(ll-1).map(i => fillRandom(mkTensor(l[i+1])));
	z = mkRange(ll-1).map(i => mkTensor(l[i+1]));
	a = mkTensor(ll-1).map(i=>lr);
	a[ll-2] = sm;
	this.variable.perceptron = p;
	this.variable.weight = w;
	this.variable.bias = b;
	this.variable.z = z;
	this.variable.activationFunction = a;
};
AI.ANN.propagation = function(inputLayer){
	let v = this.variable;
	let l = v.layer;
	let ll = l.length;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	p[0] = inputLayer;
	for(let i = 0; i < ll-1; i++){
		for(let j = 0; j < l[i+1]; j++){
			let sum = 0;
			for(let k = 0; k < l[i]; k++){
				sum += p[i][k] * w[i][j][k];
			}
			sum += b[i][j];
			z[i][j] = sum;
			p[i+1][j] = a[i](z[i][j]);
		}
	}
	this.variable.perceptron = p;
	this.variable.z = z;
	return p[ll-1];
};
AI.ANN.backPropagation = function (answer, learnRate){
	let v = this.variable;
	let l = v.layer;
	let ll = l.length;
	let p = v.perceptron;
	let w = v.weight;
	let b = v.bias;
	let z = v.z;
	let a = v.activationFunction;
	let delta = mkRange(ll-1).map(i => mkTensor(l[i+1]));
	let df = function(f,x){return (f(x+0.001)-f(x-0.001))/0.002};
	let daf = function(n,x){return df(a[n],x)};
	for(let i=0; i<l[ll-1]; i++){
		delta[ll-2][i] = 2*(p[ll-1][i]-answer[i])*daf(ll-2,z[ll-2][i]);
	}
	for(let i=0; i<l[ll-1]; i++){
		for(let j=0; j<l[ll-2]; j++){
			w[ll-2][i][j] -= learnRate * delta[ll-2][i] * p[ll-2][j];
		}
		b[ll-2][i] -= learnRate * delta[ll-2][i];
	}
	for(let i=ll-3; i>=0; i--){
		for(let j=0; j<l[i+1]; j++){
			for(let k=0; k<l[i+2]; k++){
				delta[i][j] += delta[i+1][k]*w[i+1][k][j];
			}
			delta[i][j] *= daf(i,z[i][j]);
		}
		for(let j=0; j<l[i+1]; j++){
			for(let k=0; k<l[i]; k++){
				w[i][j][k] -= learnRate * delta[i][j] * p[i][k];
			}
			b[i][j] -= learnRate * delta[i][j];
		}
	}
	this.variable.weight = w;
	this.variable.bias = b;
	let cost = 0;
	for(let i=0; i<p[ll-1].length; i++){
		cost += (p[ll-1][i]-answer[i])**2;
	}
};

AI.ANN.setLayer([2,4,1]);
// AI.ANN.variable.weight=[[[10,10],[-10,10],[-10,-10],[10,-10]],[[-10,10,-10,10]]];
// AI.ANN.variable.bias=[[0,0,0,0],[0]];

// var time = setInterval(function(){
for(let _$=0; _$<100; _$++){
	for(let n=0; n<1000; n++){
		AI.ANN.propagation([1,0]);
		AI.ANN.backPropagation([1], 0.0008);
		AI.ANN.propagation([0,1]);
		AI.ANN.backPropagation([1], 0.0008);
		AI.ANN.propagation([0,0]);
		AI.ANN.backPropagation([0], 0.0008);
		AI.ANN.propagation([1,1]);
		AI.ANN.backPropagation([0], 0.0008);
	}
}
	cl((AI.ANN.propagation([1,0])[0]-1)**2+(AI.ANN.propagation([0,1])[0]-1)**2+AI.ANN.propagation([0,0])[0]**2+AI.ANN.propagation([1,1])[0]**2);
// },1000);
// setTimeout(function(){
// 	clearInterval(time);
// 	cl('complete')
// },20000);

////////////////////////////////////////

// function learning(input, answer, learnRate, learnCount){
// 	AI.ANN.propagation(input);
// 	for(let n=0; n<learnCount; n++){
// 		AI.ANN.backPropagation(answer, learnRate);
// 	}
// }

// var pos1 = [[786,252],[656,217],[420,167],[89,70],[99,149],[116,215],[251,228],[381,207],[290,108],[331,75],[689,86],[353,41],[526,97],[725,106],[817,68],[901,98],[879,217],[792,260],[590,255],[415,220],[118,207],[282,224],[774,215],[832,215],[836,216]];
// var pos2 = [[819,787],[696,736],[540,721],[273,724],[123,738],[6,774],[194,756],[239,756],[506,744],[651,793],[436,882],[329,890],[143,896],[152,897],[560,902],[697,886],[467,814],[310,797],[569,749],[763,752],[922,790],[933,837],[910,886],[841,912],[696,928],[794,910]];

// for(let i=0; i<pos1.length; i++){
// 	learning(pos1[i],[1],0.001,10);
// }
// for(let i=0; i<pos2.length; i++){
// 	learning(pos2[i],[0],0.001,10);
// }

// document.onclick = function(e){AI.ANN.propagation([e.clientX, e.clientY]); cl(AI.ANN.variable.perceptron[2]);}


