       ŁK"	   cN×Abrain.Event:2śŢË˘      r°	=cN×A"Ĺ
g
%global_step/global_step/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
{
global_step/global_step
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ţ
global_step/global_step/AssignAssignglobal_step/global_step%global_step/global_step/initial_value*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 

global_step/global_step/readIdentityglobal_step/global_step*
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: 
c
!cur_epoch/cur_epoch/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
w
cur_epoch/cur_epoch
VariableV2*
shared_name *
dtype0*
_output_shapes
: *
	container *
shape: 
Î
cur_epoch/cur_epoch/AssignAssigncur_epoch/cur_epoch!cur_epoch/cur_epoch/initial_value*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 

cur_epoch/cur_epoch/readIdentitycur_epoch/cur_epoch*
T0*&
_class
loc:@cur_epoch/cur_epoch*
_output_shapes
: 
Q
cur_epoch/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
`
cur_epoch/addAddcur_epoch/cur_epoch/readcur_epoch/add/y*
T0*
_output_shapes
: 
°
cur_epoch/AssignAssigncur_epoch/cur_epochcur_epoch/add*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: *
use_locking(
P
PlaceholderPlaceholder*
dtype0
*
_output_shapes
:*
shape:
r
Placeholder_1Placeholder*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
shape:˙˙˙˙˙˙˙˙˙

Ą
.dense1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense1/kernel*
valueB"     

,dense1/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense1/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 

,dense1/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense1/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
ę
6dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0* 
_class
loc:@dense1/kernel*
seed2 
Ň
,dense1/kernel/Initializer/random_uniform/subSub,dense1/kernel/Initializer/random_uniform/max,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel*
_output_shapes
: 
ć
,dense1/kernel/Initializer/random_uniform/mulMul6dense1/kernel/Initializer/random_uniform/RandomUniform,dense1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel
Ř
(dense1/kernel/Initializer/random_uniformAdd,dense1/kernel/Initializer/random_uniform/mul,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:

§
dense1/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name * 
_class
loc:@dense1/kernel
Í
dense1/kernel/AssignAssigndense1/kernel(dense1/kernel/Initializer/random_uniform*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
z
dense1/kernel/readIdentitydense1/kernel* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel

dense1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense1/bias*
valueB*    

dense1/bias
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ˇ
dense1/bias/AssignAssigndense1/biasdense1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense1/bias
o
dense1/bias/readIdentitydense1/bias*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:

dense1/MatMulMatMulPlaceholder_1dense1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

dense1/BiasAddBiasAdddense1/MatMuldense1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
dense1/ReluReludense1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
.dense2/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

,dense2/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense2/kernel*
valueB
 *Ű˝*
dtype0*
_output_shapes
: 

,dense2/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense2/kernel*
valueB
 *Ű=*
dtype0*
_output_shapes
: 
é
6dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense2/kernel/Initializer/random_uniform/shape*

seed *
T0* 
_class
loc:@dense2/kernel*
seed2 *
dtype0*
_output_shapes
:	

Ň
,dense2/kernel/Initializer/random_uniform/subSub,dense2/kernel/Initializer/random_uniform/max,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
: 
ĺ
,dense2/kernel/Initializer/random_uniform/mulMul6dense2/kernel/Initializer/random_uniform/RandomUniform,dense2/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

×
(dense2/kernel/Initializer/random_uniformAdd,dense2/kernel/Initializer/random_uniform/mul,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

Ľ
dense2/kernel
VariableV2*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	

Ě
dense2/kernel/AssignAssigndense2/kernel(dense2/kernel/Initializer/random_uniform*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	
*
use_locking(
y
dense2/kernel/readIdentitydense2/kernel*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	


dense2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
loc:@dense2/bias*
valueB
*    

dense2/bias
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense2/bias*
	container *
shape:

ś
dense2/bias/AssignAssigndense2/biasdense2/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense2/bias
n
dense2/bias/readIdentitydense2/bias*
T0*
_class
loc:@dense2/bias*
_output_shapes
:


dense2/MatMulMatMuldense1/Reludense2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
transpose_b( 

dense2/BiasAddBiasAdddense2/MatMuldense2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

p
.loss/softmax_cross_entropy_with_logits_sg/RankConst*
dtype0*
_output_shapes
: *
value	B :
}
/loss/softmax_cross_entropy_with_logits_sg/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapedense2/BiasAdd*
_output_shapes
:*
T0*
out_type0
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
dtype0*
_output_shapes
: *
value	B :
¸
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
Ś
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 

0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
Ç
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapedense2/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ż
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
ź
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
Ş
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:

6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0

;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ű
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ü
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
ş
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 

7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Š
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ř
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
W
loss/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
]
loss/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
~
loss/gradients/FillFillloss/gradients/Shapeloss/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
u
+loss/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ľ
%loss/gradients/loss/Mean_grad/ReshapeReshapeloss/gradients/Fill+loss/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

#loss/gradients/loss/Mean_grad/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
ś
"loss/gradients/loss/Mean_grad/TileTile%loss/gradients/loss/Mean_grad/Reshape#loss/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

%loss/gradients/loss/Mean_grad/Shape_1Shape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
h
%loss/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
m
#loss/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
´
"loss/gradients/loss/Mean_grad/ProdProd%loss/gradients/loss/Mean_grad/Shape_1#loss/gradients/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
o
%loss/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
¸
$loss/gradients/loss/Mean_grad/Prod_1Prod%loss/gradients/loss/Mean_grad/Shape_2%loss/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
'loss/gradients/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
%loss/gradients/loss/Mean_grad/MaximumMaximum$loss/gradients/loss/Mean_grad/Prod_1'loss/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

&loss/gradients/loss/Mean_grad/floordivFloorDiv"loss/gradients/loss/Mean_grad/Prod%loss/gradients/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 

"loss/gradients/loss/Mean_grad/CastCast&loss/gradients/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ś
%loss/gradients/loss/Mean_grad/truedivRealDiv"loss/gradients/loss/Mean_grad/Tile"loss/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0

Oloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape%loss/gradients/loss/Mean_grad/truedivMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Lloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ł
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeLloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ä
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Í
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙
ˇ
Jloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeNloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0

Closs/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulJloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ŕ
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpB^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulD^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1

Vloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityAloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulO^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Xloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityCloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1O^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*V
_classL
JHloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Kloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
˝
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeVloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyKloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
Tshape0
Č
.loss/gradients/dense2/BiasAdd_grad/BiasAddGradBiasAddGradMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

ź
3loss/gradients/dense2/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense2/BiasAdd_grad/BiasAddGradN^loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
ŕ
;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencyIdentityMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense2/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*
T0*A
_class7
53loc:@loss/gradients/dense2/BiasAdd_grad/BiasAddGrad
Ü
(loss/gradients/dense2/MatMul_grad/MatMulMatMul;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencydense2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Î
*loss/gradients/dense2/MatMul_grad/MatMul_1MatMuldense1/Relu;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	
*
transpose_a(*
transpose_b( *
T0

2loss/gradients/dense2/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense2/MatMul_grad/MatMul+^loss/gradients/dense2/MatMul_grad/MatMul_1

:loss/gradients/dense2/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense2/MatMul_grad/MatMul3^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense2/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense2/MatMul_grad/MatMul_13^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
_output_shapes
:	
*
T0*=
_class3
1/loc:@loss/gradients/dense2/MatMul_grad/MatMul_1
°
(loss/gradients/dense1/Relu_grad/ReluGradReluGrad:loss/gradients/dense2/MatMul_grad/tuple/control_dependencydense1/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
.loss/gradients/dense1/BiasAdd_grad/BiasAddGradBiasAddGrad(loss/gradients/dense1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3loss/gradients/dense1/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense1/BiasAdd_grad/BiasAddGrad)^loss/gradients/dense1/Relu_grad/ReluGrad

;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/Relu_grad/ReluGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense1/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ü
(loss/gradients/dense1/MatMul_grad/MatMulMatMul;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencydense1/kernel/read*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
Ń
*loss/gradients/dense1/MatMul_grad/MatMul_1MatMulPlaceholder_1;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
*
transpose_a(*
transpose_b( 

2loss/gradients/dense1/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense1/MatMul_grad/MatMul+^loss/gradients/dense1/MatMul_grad/MatMul_1

:loss/gradients/dense1/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/MatMul_grad/MatMul3^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense1/MatMul_grad/MatMul_13^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense1/MatMul_grad/MatMul_1* 
_output_shapes
:


loss/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@dense1/bias*
valueB
 *fff?

loss/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@dense1/bias*
	container 
˝
loss/beta1_power/AssignAssignloss/beta1_powerloss/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
t
loss/beta1_power/readIdentityloss/beta1_power*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 

loss/beta2_power/initial_valueConst*
_class
loc:@dense1/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

loss/beta2_power
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
˝
loss/beta2_power/AssignAssignloss/beta2_powerloss/beta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense1/bias
t
loss/beta2_power/readIdentityloss/beta2_power*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
§
4dense1/kernel/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:

*dense1/kernel/Adam/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
í
$dense1/kernel/Adam/Initializer/zerosFill4dense1/kernel/Adam/Initializer/zeros/shape_as_tensor*dense1/kernel/Adam/Initializer/zeros/Const*
T0* 
_class
loc:@dense1/kernel*

index_type0* 
_output_shapes
:

Ź
dense1/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name * 
_class
loc:@dense1/kernel*
	container 
Ó
dense1/kernel/Adam/AssignAssigndense1/kernel/Adam$dense1/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0* 
_class
loc:@dense1/kernel

dense1/kernel/Adam/readIdentitydense1/kernel/Adam* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel
Š
6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:

,dense1/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ó
&dense1/kernel/Adam_1/Initializer/zerosFill6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense1/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@dense1/kernel*

index_type0* 
_output_shapes
:

Ž
dense1/kernel/Adam_1
VariableV2* 
_class
loc:@dense1/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ů
dense1/kernel/Adam_1/AssignAssigndense1/kernel/Adam_1&dense1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:


dense1/kernel/Adam_1/readIdentitydense1/kernel/Adam_1*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:


"dense1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense1/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:
Ć
dense1/bias/Adam/AssignAssigndense1/bias/Adam"dense1/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
y
dense1/bias/Adam/readIdentitydense1/bias/Adam*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:

$dense1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense1/bias*
valueB*    
 
dense1/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ě
dense1/bias/Adam_1/AssignAssigndense1/bias/Adam_1$dense1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:
}
dense1/bias/Adam_1/readIdentitydense1/bias/Adam_1*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:
§
4dense2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense2/kernel*
valueB"   
   

*dense2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense2/kernel*
valueB
 *    
ě
$dense2/kernel/Adam/Initializer/zerosFill4dense2/kernel/Adam/Initializer/zeros/shape_as_tensor*dense2/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	
*
T0* 
_class
loc:@dense2/kernel*

index_type0
Ş
dense2/kernel/Adam
VariableV2* 
_class
loc:@dense2/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name 
Ň
dense2/kernel/Adam/AssignAssigndense2/kernel/Adam$dense2/kernel/Adam/Initializer/zeros*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	
*
use_locking(

dense2/kernel/Adam/readIdentitydense2/kernel/Adam*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

Š
6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

,dense2/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ň
&dense2/kernel/Adam_1/Initializer/zerosFill6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense2/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	
*
T0* 
_class
loc:@dense2/kernel*

index_type0
Ź
dense2/kernel/Adam_1
VariableV2* 
_class
loc:@dense2/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name 
Ř
dense2/kernel/Adam_1/AssignAssigndense2/kernel/Adam_1&dense2/kernel/Adam_1/Initializer/zeros*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	
*
use_locking(

dense2/kernel/Adam_1/readIdentitydense2/kernel/Adam_1*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	


"dense2/bias/Adam/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense2/bias/Adam
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense2/bias*
	container 
Ĺ
dense2/bias/Adam/AssignAssigndense2/bias/Adam"dense2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

x
dense2/bias/Adam/readIdentitydense2/bias/Adam*
T0*
_class
loc:@dense2/bias*
_output_shapes
:


$dense2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense2/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@dense2/bias*
	container *
shape:
*
dtype0*
_output_shapes
:

Ë
dense2/bias/Adam_1/AssignAssigndense2/bias/Adam_1$dense2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

|
dense2/bias/Adam_1/readIdentitydense2/bias/Adam_1*
T0*
_class
loc:@dense2/bias*
_output_shapes
:

\
loss/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o:
T
loss/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
T
loss/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
V
loss/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

(loss/Adam/update_dense1/kernel/ApplyAdam	ApplyAdamdense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1*
T0* 
_class
loc:@dense1/kernel*
use_nesterov( * 
_output_shapes
:
*
use_locking( 

&loss/Adam/update_dense1/bias/ApplyAdam	ApplyAdamdense1/biasdense1/bias/Adamdense1/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense1/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 

(loss/Adam/update_dense2/kernel/ApplyAdam	ApplyAdamdense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@dense2/kernel*
use_nesterov( *
_output_shapes
:	


&loss/Adam/update_dense2/bias/ApplyAdam	ApplyAdamdense2/biasdense2/bias/Adamdense2/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense2/bias*
use_nesterov( *
_output_shapes
:

Ľ
loss/Adam/mulMulloss/beta1_power/readloss/Adam/beta1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
Ľ
loss/Adam/AssignAssignloss/beta1_powerloss/Adam/mul*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
§
loss/Adam/mul_1Mulloss/beta2_power/readloss/Adam/beta2'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
Š
loss/Adam/Assign_1Assignloss/beta2_powerloss/Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
č
loss/Adam/updateNoOp^loss/Adam/Assign^loss/Adam/Assign_1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam

loss/Adam/valueConst^loss/Adam/update**
_class 
loc:@global_step/global_step*
value	B :*
dtype0*
_output_shapes
: 
 
	loss/Adam	AssignAddglobal_step/global_steploss/Adam/value*
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: *
use_locking( 
W
loss/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

loss/ArgMaxArgMaxdense2/BiasAddloss/ArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Y
loss/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

loss/ArgMax_1ArgMaxPlaceholder_2loss/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
]

loss/EqualEqualloss/ArgMaxloss/ArgMax_1*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
	loss/CastCast
loss/Equal*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
V
loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
j
loss/Mean_1Mean	loss/Castloss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/SaveV2/tensor_namesConst*ż
valueľB˛Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ł
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicescur_epoch/cur_epochdense1/biasdense1/bias/Adamdense1/bias/Adam_1dense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1dense2/biasdense2/bias/Adamdense2/bias/Adam_1dense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1global_step/global_steploss/beta1_powerloss/beta2_power*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*ż
valueľB˛Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
ę
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2
Ź
save/AssignAssigncur_epoch/cur_epochsave/RestoreV2*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
Ľ
save/Assign_1Assigndense1/biassave/RestoreV2:1*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense1/bias
Ş
save/Assign_2Assigndense1/bias/Adamsave/RestoreV2:2*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense1/bias
Ź
save/Assign_3Assigndense1/bias/Adam_1save/RestoreV2:3*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense1/bias
Ž
save/Assign_4Assigndense1/kernelsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

ł
save/Assign_5Assigndense1/kernel/Adamsave/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

ľ
save/Assign_6Assigndense1/kernel/Adam_1save/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

¤
save/Assign_7Assigndense2/biassave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Š
save/Assign_8Assigndense2/bias/Adamsave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Ť
save/Assign_9Assigndense2/bias/Adam_1save/RestoreV2:9*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Ż
save/Assign_10Assigndense2/kernelsave/RestoreV2:10*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	
*
use_locking(
´
save/Assign_11Assigndense2/kernel/Adamsave/RestoreV2:11*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	
*
use_locking(
ś
save/Assign_12Assigndense2/kernel/Adam_1save/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	

ş
save/Assign_13Assignglobal_step/global_stepsave/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 
§
save/Assign_14Assignloss/beta1_powersave/RestoreV2:14*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
§
save/Assign_15Assignloss/beta2_powersave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"şšĚŻóš      ď[č	§@?cN×AJćó
ĺÎ
:
Add
x"T
y"T
z"T"
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
2
StopGradient

input"T
output"T"	
Ttype
:
Sub
x"T
y"T
z"T"
Ttype:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.10.02
b'unknown'Ĺ
g
%global_step/global_step/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
{
global_step/global_step
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Ţ
global_step/global_step/AssignAssignglobal_step/global_step%global_step/global_step/initial_value*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 

global_step/global_step/readIdentityglobal_step/global_step*
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: 
c
!cur_epoch/cur_epoch/initial_valueConst*
_output_shapes
: *
value	B : *
dtype0
w
cur_epoch/cur_epoch
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Î
cur_epoch/cur_epoch/AssignAssigncur_epoch/cur_epoch!cur_epoch/cur_epoch/initial_value*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 

cur_epoch/cur_epoch/readIdentitycur_epoch/cur_epoch*
T0*&
_class
loc:@cur_epoch/cur_epoch*
_output_shapes
: 
Q
cur_epoch/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
`
cur_epoch/addAddcur_epoch/cur_epoch/readcur_epoch/add/y*
T0*
_output_shapes
: 
°
cur_epoch/AssignAssigncur_epoch/cur_epochcur_epoch/add*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch
P
PlaceholderPlaceholder*
dtype0
*
_output_shapes
:*
shape:
r
Placeholder_1Placeholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
shape:˙˙˙˙˙˙˙˙˙

Ą
.dense1/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:

,dense1/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense1/kernel*
valueB
 *HY˝*
dtype0*
_output_shapes
: 

,dense1/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense1/kernel*
valueB
 *HY=*
dtype0*
_output_shapes
: 
ę
6dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0* 
_class
loc:@dense1/kernel*
seed2 
Ň
,dense1/kernel/Initializer/random_uniform/subSub,dense1/kernel/Initializer/random_uniform/max,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel*
_output_shapes
: 
ć
,dense1/kernel/Initializer/random_uniform/mulMul6dense1/kernel/Initializer/random_uniform/RandomUniform,dense1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel
Ř
(dense1/kernel/Initializer/random_uniformAdd,dense1/kernel/Initializer/random_uniform/mul,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:

§
dense1/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name * 
_class
loc:@dense1/kernel*
	container *
shape:

Í
dense1/kernel/AssignAssigndense1/kernel(dense1/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

z
dense1/kernel/readIdentitydense1/kernel*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:


dense1/bias/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueB*    *
dtype0*
_output_shapes	
:

dense1/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:
ˇ
dense1/bias/AssignAssigndense1/biasdense1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:
o
dense1/bias/readIdentitydense1/bias*
_output_shapes	
:*
T0*
_class
loc:@dense1/bias

dense1/MatMulMatMulPlaceholder_1dense1/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense1/BiasAddBiasAdddense1/MatMuldense1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
V
dense1/ReluReludense1/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
.dense2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0

,dense2/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense2/kernel*
valueB
 *Ű˝*
dtype0*
_output_shapes
: 

,dense2/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense2/kernel*
valueB
 *Ű=*
dtype0*
_output_shapes
: 
é
6dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	
*

seed *
T0* 
_class
loc:@dense2/kernel*
seed2 
Ň
,dense2/kernel/Initializer/random_uniform/subSub,dense2/kernel/Initializer/random_uniform/max,dense2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@dense2/kernel
ĺ
,dense2/kernel/Initializer/random_uniform/mulMul6dense2/kernel/Initializer/random_uniform/RandomUniform,dense2/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

×
(dense2/kernel/Initializer/random_uniformAdd,dense2/kernel/Initializer/random_uniform/mul,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

Ľ
dense2/kernel
VariableV2* 
_class
loc:@dense2/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name 
Ě
dense2/kernel/AssignAssigndense2/kernel(dense2/kernel/Initializer/random_uniform*
_output_shapes
:	
*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(
y
dense2/kernel/readIdentitydense2/kernel*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	


dense2/bias/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense2/bias
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense2/bias*
	container 
ś
dense2/bias/AssignAssigndense2/biasdense2/bias/Initializer/zeros*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0
n
dense2/bias/readIdentitydense2/bias*
T0*
_class
loc:@dense2/bias*
_output_shapes
:


dense2/MatMulMatMuldense1/Reludense2/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
transpose_a( *
transpose_b( 

dense2/BiasAddBiasAdddense2/MatMuldense2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_2*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

p
.loss/softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0*
_output_shapes
: 
}
/loss/softmax_cross_entropy_with_logits_sg/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 

1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
¸
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0
Ś
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N*
_output_shapes
:
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:

/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
Index0*
T0*
_output_shapes
:

9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
Ç
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapedense2/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Ż
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
ź
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
_output_shapes
: *
T0
Ş
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*

axis *
N*
_output_shapes
:*
T0

6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:

;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ű
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
ü
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
ş
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 

7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
Š
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:

1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ř
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:

	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
W
loss/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
]
loss/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
~
loss/gradients/FillFillloss/gradients/Shapeloss/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
u
+loss/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ľ
%loss/gradients/loss/Mean_grad/ReshapeReshapeloss/gradients/Fill+loss/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

#loss/gradients/loss/Mean_grad/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
out_type0*
_output_shapes
:*
T0
ś
"loss/gradients/loss/Mean_grad/TileTile%loss/gradients/loss/Mean_grad/Reshape#loss/gradients/loss/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

%loss/gradients/loss/Mean_grad/Shape_1Shape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
h
%loss/gradients/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
m
#loss/gradients/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
´
"loss/gradients/loss/Mean_grad/ProdProd%loss/gradients/loss/Mean_grad/Shape_1#loss/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
o
%loss/gradients/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
¸
$loss/gradients/loss/Mean_grad/Prod_1Prod%loss/gradients/loss/Mean_grad/Shape_2%loss/gradients/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
'loss/gradients/loss/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
 
%loss/gradients/loss/Mean_grad/MaximumMaximum$loss/gradients/loss/Mean_grad/Prod_1'loss/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

&loss/gradients/loss/Mean_grad/floordivFloorDiv"loss/gradients/loss/Mean_grad/Prod%loss/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0

"loss/gradients/loss/Mean_grad/CastCast&loss/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ś
%loss/gradients/loss/Mean_grad/truedivRealDiv"loss/gradients/loss/Mean_grad/Tile"loss/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0

Oloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape%loss/gradients/loss/Mean_grad/truedivMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Lloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ł
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeLloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0
ú
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ä
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Í
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ˇ
Jloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeNloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0

Closs/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulJloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
ŕ
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpB^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulD^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1

Vloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityAloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulO^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Xloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityCloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1O^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*V
_classL
JHloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Kloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
˝
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeVloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyKloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Č
.loss/gradients/dense2/BiasAdd_grad/BiasAddGradBiasAddGradMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
data_formatNHWC*
_output_shapes
:
*
T0
ź
3loss/gradients/dense2/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense2/BiasAdd_grad/BiasAddGradN^loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
ŕ
;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencyIdentityMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*`
_classV
TRloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape

=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense2/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

Ü
(loss/gradients/dense2/MatMul_grad/MatMulMatMul;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencydense2/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(
Î
*loss/gradients/dense2/MatMul_grad/MatMul_1MatMuldense1/Relu;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	
*
transpose_a(*
transpose_b( 

2loss/gradients/dense2/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense2/MatMul_grad/MatMul+^loss/gradients/dense2/MatMul_grad/MatMul_1

:loss/gradients/dense2/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense2/MatMul_grad/MatMul3^loss/gradients/dense2/MatMul_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*;
_class1
/-loc:@loss/gradients/dense2/MatMul_grad/MatMul

<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense2/MatMul_grad/MatMul_13^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense2/MatMul_grad/MatMul_1*
_output_shapes
:	

°
(loss/gradients/dense1/Relu_grad/ReluGradReluGrad:loss/gradients/dense2/MatMul_grad/tuple/control_dependencydense1/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
¤
.loss/gradients/dense1/BiasAdd_grad/BiasAddGradBiasAddGrad(loss/gradients/dense1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:

3loss/gradients/dense1/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense1/BiasAdd_grad/BiasAddGrad)^loss/gradients/dense1/Relu_grad/ReluGrad

;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/Relu_grad/ReluGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense1/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
Ü
(loss/gradients/dense1/MatMul_grad/MatMulMatMul;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencydense1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b(*
T0
Ń
*loss/gradients/dense1/MatMul_grad/MatMul_1MatMulPlaceholder_1;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a(

2loss/gradients/dense1/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense1/MatMul_grad/MatMul+^loss/gradients/dense1/MatMul_grad/MatMul_1

:loss/gradients/dense1/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/MatMul_grad/MatMul3^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense1/MatMul_grad/MatMul_13^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense1/MatMul_grad/MatMul_1* 
_output_shapes
:


loss/beta1_power/initial_valueConst*
_class
loc:@dense1/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 

loss/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@dense1/bias*
	container *
shape: 
˝
loss/beta1_power/AssignAssignloss/beta1_powerloss/beta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
t
loss/beta1_power/readIdentityloss/beta1_power*
_output_shapes
: *
T0*
_class
loc:@dense1/bias

loss/beta2_power/initial_valueConst*
_class
loc:@dense1/bias*
valueB
 *wž?*
dtype0*
_output_shapes
: 

loss/beta2_power
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
˝
loss/beta2_power/AssignAssignloss/beta2_powerloss/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
t
loss/beta2_power/readIdentityloss/beta2_power*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
§
4dense1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense1/kernel*
valueB"     

*dense1/kernel/Adam/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
í
$dense1/kernel/Adam/Initializer/zerosFill4dense1/kernel/Adam/Initializer/zeros/shape_as_tensor*dense1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel*

index_type0
Ź
dense1/kernel/Adam
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name * 
_class
loc:@dense1/kernel*
	container 
Ó
dense1/kernel/Adam/AssignAssigndense1/kernel/Adam$dense1/kernel/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:


dense1/kernel/Adam/readIdentitydense1/kernel/Adam*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:

Š
6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:

,dense1/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ó
&dense1/kernel/Adam_1/Initializer/zerosFill6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense1/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0* 
_class
loc:@dense1/kernel*

index_type0
Ž
dense1/kernel/Adam_1
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name * 
_class
loc:@dense1/kernel*
	container 
Ů
dense1/kernel/Adam_1/AssignAssigndense1/kernel/Adam_1&dense1/kernel/Adam_1/Initializer/zeros*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

dense1/kernel/Adam_1/readIdentitydense1/kernel/Adam_1*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:


"dense1/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense1/bias*
valueB*    

dense1/bias/Adam
VariableV2*
_class
loc:@dense1/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ć
dense1/bias/Adam/AssignAssigndense1/bias/Adam"dense1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:
y
dense1/bias/Adam/readIdentitydense1/bias/Adam*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:

$dense1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueB*    *
dtype0*
_output_shapes	
:
 
dense1/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *
_class
loc:@dense1/bias*
	container 
Ě
dense1/bias/Adam_1/AssignAssigndense1/bias/Adam_1$dense1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense1/bias
}
dense1/bias/Adam_1/readIdentitydense1/bias/Adam_1*
_output_shapes	
:*
T0*
_class
loc:@dense1/bias
§
4dense2/kernel/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

*dense2/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense2/kernel*
valueB
 *    
ě
$dense2/kernel/Adam/Initializer/zerosFill4dense2/kernel/Adam/Initializer/zeros/shape_as_tensor*dense2/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	
*
T0* 
_class
loc:@dense2/kernel*

index_type0
Ş
dense2/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	
*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	

Ň
dense2/kernel/Adam/AssignAssigndense2/kernel/Adam$dense2/kernel/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	


dense2/kernel/Adam/readIdentitydense2/kernel/Adam*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	

Š
6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:

,dense2/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ň
&dense2/kernel/Adam_1/Initializer/zerosFill6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense2/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@dense2/kernel*

index_type0*
_output_shapes
:	

Ź
dense2/kernel/Adam_1
VariableV2* 
_class
loc:@dense2/kernel*
	container *
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name 
Ř
dense2/kernel/Adam_1/AssignAssigndense2/kernel/Adam_1&dense2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	


dense2/kernel/Adam_1/readIdentitydense2/kernel/Adam_1*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	


"dense2/bias/Adam/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense2/bias/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense2/bias*
	container *
shape:

Ĺ
dense2/bias/Adam/AssignAssigndense2/bias/Adam"dense2/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
x
dense2/bias/Adam/readIdentitydense2/bias/Adam*
_output_shapes
:
*
T0*
_class
loc:@dense2/bias

$dense2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense2/bias/Adam_1
VariableV2*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
loc:@dense2/bias
Ë
dense2/bias/Adam_1/AssignAssigndense2/bias/Adam_1$dense2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

|
dense2/bias/Adam_1/readIdentitydense2/bias/Adam_1*
_output_shapes
:
*
T0*
_class
loc:@dense2/bias
\
loss/Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
T
loss/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
T
loss/Adam/beta2Const*
valueB
 *wž?*
dtype0*
_output_shapes
: 
V
loss/Adam/epsilonConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 

(loss/Adam/update_dense1/kernel/ApplyAdam	ApplyAdamdense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
*
use_locking( *
T0* 
_class
loc:@dense1/kernel

&loss/Adam/update_dense1/bias/ApplyAdam	ApplyAdamdense1/biasdense1/bias/Adamdense1/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
use_nesterov( *
_output_shapes	
:

(loss/Adam/update_dense2/kernel/ApplyAdam	ApplyAdamdense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1*
T0* 
_class
loc:@dense2/kernel*
use_nesterov( *
_output_shapes
:	
*
use_locking( 

&loss/Adam/update_dense2/bias/ApplyAdam	ApplyAdamdense2/biasdense2/bias/Adamdense2/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense2/bias*
use_nesterov( *
_output_shapes
:

Ľ
loss/Adam/mulMulloss/beta1_power/readloss/Adam/beta1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
Ľ
loss/Adam/AssignAssignloss/beta1_powerloss/Adam/mul*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
§
loss/Adam/mul_1Mulloss/beta2_power/readloss/Adam/beta2'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
Š
loss/Adam/Assign_1Assignloss/beta2_powerloss/Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
č
loss/Adam/updateNoOp^loss/Adam/Assign^loss/Adam/Assign_1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam

loss/Adam/valueConst^loss/Adam/update**
_class 
loc:@global_step/global_step*
value	B :*
dtype0*
_output_shapes
: 
 
	loss/Adam	AssignAddglobal_step/global_steploss/Adam/value*
use_locking( *
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: 
W
loss/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

loss/ArgMaxArgMaxdense2/BiasAddloss/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Y
loss/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

loss/ArgMax_1ArgMaxPlaceholder_2loss/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
]

loss/EqualEqualloss/ArgMaxloss/ArgMax_1*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
	loss/CastCast
loss/Equal*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

V
loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
j
loss/Mean_1Mean	loss/Castloss/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/SaveV2/tensor_namesConst*ż
valueľB˛Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
Ł
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicescur_epoch/cur_epochdense1/biasdense1/bias/Adamdense1/bias/Adam_1dense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1dense2/biasdense2/bias/Adamdense2/bias/Adam_1dense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1global_step/global_steploss/beta1_powerloss/beta2_power*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 

save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*ż
valueľB˛Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
ę
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2
Ź
save/AssignAssigncur_epoch/cur_epochsave/RestoreV2*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
Ľ
save/Assign_1Assigndense1/biassave/RestoreV2:1*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ş
save/Assign_2Assigndense1/bias/Adamsave/RestoreV2:2*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ź
save/Assign_3Assigndense1/bias/Adam_1save/RestoreV2:3*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ž
save/Assign_4Assigndense1/kernelsave/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

ł
save/Assign_5Assigndense1/kernel/Adamsave/RestoreV2:5*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0* 
_class
loc:@dense1/kernel
ľ
save/Assign_6Assigndense1/kernel/Adam_1save/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:

¤
save/Assign_7Assigndense2/biassave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Š
save/Assign_8Assigndense2/bias/Adamsave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Ť
save/Assign_9Assigndense2/bias/Adam_1save/RestoreV2:9*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense2/bias
Ż
save/Assign_10Assigndense2/kernelsave/RestoreV2:10*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	

´
save/Assign_11Assigndense2/kernel/Adamsave/RestoreV2:11*
validate_shape(*
_output_shapes
:	
*
use_locking(*
T0* 
_class
loc:@dense2/kernel
ś
save/Assign_12Assigndense2/kernel/Adam_1save/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	

ş
save/Assign_13Assignglobal_step/global_stepsave/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 
§
save/Assign_14Assignloss/beta1_powersave/RestoreV2:14*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense1/bias
§
save/Assign_15Assignloss/beta2_powersave/RestoreV2:15*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: *
use_locking(

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""
train_op

	loss/Adam"¤
	variables

global_step/global_step:0global_step/global_step/Assignglobal_step/global_step/read:02'global_step/global_step/initial_value:0
t
cur_epoch/cur_epoch:0cur_epoch/cur_epoch/Assigncur_epoch/cur_epoch/read:02#cur_epoch/cur_epoch/initial_value:0
k
dense1/kernel:0dense1/kernel/Assigndense1/kernel/read:02*dense1/kernel/Initializer/random_uniform:08
Z
dense1/bias:0dense1/bias/Assigndense1/bias/read:02dense1/bias/Initializer/zeros:08
k
dense2/kernel:0dense2/kernel/Assigndense2/kernel/read:02*dense2/kernel/Initializer/random_uniform:08
Z
dense2/bias:0dense2/bias/Assigndense2/bias/read:02dense2/bias/Initializer/zeros:08
h
loss/beta1_power:0loss/beta1_power/Assignloss/beta1_power/read:02 loss/beta1_power/initial_value:0
h
loss/beta2_power:0loss/beta2_power/Assignloss/beta2_power/read:02 loss/beta2_power/initial_value:0
t
dense1/kernel/Adam:0dense1/kernel/Adam/Assigndense1/kernel/Adam/read:02&dense1/kernel/Adam/Initializer/zeros:0
|
dense1/kernel/Adam_1:0dense1/kernel/Adam_1/Assigndense1/kernel/Adam_1/read:02(dense1/kernel/Adam_1/Initializer/zeros:0
l
dense1/bias/Adam:0dense1/bias/Adam/Assigndense1/bias/Adam/read:02$dense1/bias/Adam/Initializer/zeros:0
t
dense1/bias/Adam_1:0dense1/bias/Adam_1/Assigndense1/bias/Adam_1/read:02&dense1/bias/Adam_1/Initializer/zeros:0
t
dense2/kernel/Adam:0dense2/kernel/Adam/Assigndense2/kernel/Adam/read:02&dense2/kernel/Adam/Initializer/zeros:0
|
dense2/kernel/Adam_1:0dense2/kernel/Adam_1/Assigndense2/kernel/Adam_1/read:02(dense2/kernel/Adam_1/Initializer/zeros:0
l
dense2/bias/Adam:0dense2/bias/Adam/Assigndense2/bias/Adam/read:02$dense2/bias/Adam/Initializer/zeros:0
t
dense2/bias/Adam_1:0dense2/bias/Adam_1/Assigndense2/bias/Adam_1/read:02&dense2/bias/Adam_1/Initializer/zeros:0"­
trainable_variables
k
dense1/kernel:0dense1/kernel/Assigndense1/kernel/read:02*dense1/kernel/Initializer/random_uniform:08
Z
dense1/bias:0dense1/bias/Assigndense1/bias/read:02dense1/bias/Initializer/zeros:08
k
dense2/kernel:0dense2/kernel/Assigndense2/kernel/read:02*dense2/kernel/Initializer/random_uniform:08
Z
dense2/bias:0dense2/bias/Assigndense2/bias/read:02dense2/bias/Initializer/zeros:08uVsĚ       ČÁ	çÄUcN×A
*

loss_2níBÖ_ŽE       	<ŘUcN×A
*

acc_1    ŽhJö       ČÁ	ýecN×A*

loss_2óBe^6       	š fcN×A*

acc_1    ĚlÇę       ČÁ	ČtcN×A*

loss_2ÂáĘB%đů       	 
tcN×A*

acc_1    Ś       ČÁ	Ď/cN×A(*

loss_2˘C;Ü_ý       	H1cN×A(*

acc_1    ě.::       ČÁ	XucN×A2*

loss_2Ju-C9'.Č       	vcN×A2*

acc_1    ÓoŘ       ČÁ	cN×A<*

loss_2z8TCëĺ4Ś       	çcN×A<*

acc_1    ŕ~H       ČÁ	FŽcN×AF*

loss_2x|CĂ¸       	9ŽcN×AF*

acc_1    Ó()       ČÁ	g˝cN×AP*

loss_2+Cn°{Ó       	bh˝cN×AP*

acc_1    żM1ř       ČÁ	<JĚcN×AZ*

loss_2Ň¨C7hm       	{KĚcN×AZ*

acc_1    ť´Őę       ČÁ	`äcN×Ad*

loss_2YžC4c       	NbäcN×Ad*

acc_1    ś~ś       ČÁ	˛+ňcN×An*

loss_2ŚwÔCáW       	ţ,ňcN×An*

acc_1    Î