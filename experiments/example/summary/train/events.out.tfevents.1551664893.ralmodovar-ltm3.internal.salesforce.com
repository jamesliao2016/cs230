       £K"	  @њ!„Abrain.Event:2g"ґХҐ      rГ∞	Ёsњ!„A"И≈
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
ё
global_step/global_step/AssignAssignglobal_step/global_step%global_step/global_step/initial_value*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 
О
global_step/global_step/readIdentityglobal_step/global_step*
_output_shapes
: *
T0**
_class 
loc:@global_step/global_step
c
!cur_epoch/cur_epoch/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
w
cur_epoch/cur_epoch
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
ќ
cur_epoch/cur_epoch/AssignAssigncur_epoch/cur_epoch!cur_epoch/cur_epoch/initial_value*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
В
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
∞
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
dtype0*(
_output_shapes
:€€€€€€€€€Р*
shape:€€€€€€€€€Р
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€
*
shape:€€€€€€€€€

°
.dense1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense1/kernel*
valueB"     
У
,dense1/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense1/kernel*
valueB
 *HYЛљ*
dtype0*
_output_shapes
: 
У
,dense1/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense1/kernel*
valueB
 *HYЛ=*
dtype0*
_output_shapes
: 
к
6dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense1/kernel/Initializer/random_uniform/shape*
T0* 
_class
loc:@dense1/kernel*
seed2 *
dtype0* 
_output_shapes
:
РА*

seed 
“
,dense1/kernel/Initializer/random_uniform/subSub,dense1/kernel/Initializer/random_uniform/max,dense1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@dense1/kernel
ж
,dense1/kernel/Initializer/random_uniform/mulMul6dense1/kernel/Initializer/random_uniform/RandomUniform,dense1/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
Ў
(dense1/kernel/Initializer/random_uniformAdd,dense1/kernel/Initializer/random_uniform/mul,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
І
dense1/kernel
VariableV2*
dtype0* 
_output_shapes
:
РА*
shared_name * 
_class
loc:@dense1/kernel*
	container *
shape:
РА
Ќ
dense1/kernel/AssignAssigndense1/kernel(dense1/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА
z
dense1/kernel/readIdentitydense1/kernel*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
М
dense1/bias/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Щ
dense1/bias
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@dense1/bias
Ј
dense1/bias/AssignAssigndense1/biasdense1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А
o
dense1/bias/readIdentitydense1/bias*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
У
dense1/MatMulMatMulPlaceholder_1dense1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( 
Д
dense1/BiasAddBiasAdddense1/MatMuldense1/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
V
dense1/ReluReludense1/BiasAdd*(
_output_shapes
:€€€€€€€€€А*
T0
°
.dense2/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
У
,dense2/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense2/kernel*
valueB
 *УСџљ*
dtype0*
_output_shapes
: 
У
,dense2/kernel/Initializer/random_uniform/maxConst* 
_class
loc:@dense2/kernel*
valueB
 *УСџ=*
dtype0*
_output_shapes
: 
й
6dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А
*

seed *
T0* 
_class
loc:@dense2/kernel*
seed2 
“
,dense2/kernel/Initializer/random_uniform/subSub,dense2/kernel/Initializer/random_uniform/max,dense2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0* 
_class
loc:@dense2/kernel
е
,dense2/kernel/Initializer/random_uniform/mulMul6dense2/kernel/Initializer/random_uniform/RandomUniform,dense2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А
*
T0* 
_class
loc:@dense2/kernel
„
(dense2/kernel/Initializer/random_uniformAdd,dense2/kernel/Initializer/random_uniform/mul,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

•
dense2/kernel
VariableV2*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	А
*
dtype0*
_output_shapes
:	А

ћ
dense2/kernel/AssignAssigndense2/kernel(dense2/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

y
dense2/kernel/readIdentitydense2/kernel*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

К
dense2/bias/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:

Ч
dense2/bias
VariableV2*
_class
loc:@dense2/bias*
	container *
shape:
*
dtype0*
_output_shapes
:
*
shared_name 
ґ
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

Р
dense2/MatMulMatMuldense1/Reludense2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
Г
dense2/BiasAddBiasAdddense2/MatMuldense2/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
*
T0
П
>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_2*'
_output_shapes
:€€€€€€€€€
*
T0
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
1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapedense2/BiasAdd*
_output_shapes
:*
T0*
out_type0
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Є
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
T0*
_output_shapes
: 
¶
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
N*
_output_shapes
:*
T0*

axis 
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
К
/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0
М
9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Щ
0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
«
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapedense2/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
ѓ
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
_output_shapes
:*
T0*
out_type0
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
Љ
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
™
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
А
6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Р
1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
Index0*
T0*
_output_shapes
:
О
;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
°
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
ы
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
ь
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*
T0*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
dtype0*
_output_shapes
: *
value	B :
Ї
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
Б
7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
©
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
N*
_output_shapes
:*
T0*

axis 
О
1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ў
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Р
	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
W
loss/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
]
loss/gradients/grad_ys_0Const*
valueB
 *  А?*
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
•
%loss/gradients/loss/Mean_grad/ReshapeReshapeloss/gradients/Fill+loss/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Ц
#loss/gradients/loss/Mean_grad/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
ґ
"loss/gradients/loss/Mean_grad/TileTile%loss/gradients/loss/Mean_grad/Reshape#loss/gradients/loss/Mean_grad/Shape*#
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
Ш
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
#loss/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
і
"loss/gradients/loss/Mean_grad/ProdProd%loss/gradients/loss/Mean_grad/Shape_1#loss/gradients/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
o
%loss/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Є
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
†
%loss/gradients/loss/Mean_grad/MaximumMaximum$loss/gradients/loss/Mean_grad/Prod_1'loss/gradients/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Ю
&loss/gradients/loss/Mean_grad/floordivFloorDiv"loss/gradients/loss/Mean_grad/Prod%loss/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
В
"loss/gradients/loss/Mean_grad/CastCast&loss/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
¶
%loss/gradients/loss/Mean_grad/truedivRealDiv"loss/gradients/loss/Mean_grad/Tile"loss/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
ґ
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
T0*
out_type0*
_output_shapes
:
М
Oloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape%loss/gradients/loss/Mean_grad/truedivMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
О
loss/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ч
Lloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
≥
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeLloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*

Tdim0*
T0
ъ
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
Ќ
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Щ
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ј
Jloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeNloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*
T0*'
_output_shapes
:€€€€€€€€€*

Tdim0
Ф
Closs/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulJloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
а
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpB^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulD^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1
З
Vloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityAloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulO^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Н
Xloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityCloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1O^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*V
_classL
JHloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Щ
Kloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
љ
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeVloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyKloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

»
.loss/gradients/dense2/BiasAdd_grad/BiasAddGradBiasAddGradMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

Љ
3loss/gradients/dense2/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense2/BiasAdd_grad/BiasAddGradN^loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
а
;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencyIdentityMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*'
_output_shapes
:€€€€€€€€€

Ч
=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense2/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:

№
(loss/gradients/dense2/MatMul_grad/MatMulMatMul;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencydense2/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b(
ќ
*loss/gradients/dense2/MatMul_grad/MatMul_1MatMuldense1/Relu;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	А
*
transpose_a(*
transpose_b( 
Т
2loss/gradients/dense2/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense2/MatMul_grad/MatMul+^loss/gradients/dense2/MatMul_grad/MatMul_1
Х
:loss/gradients/dense2/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense2/MatMul_grad/MatMul3^loss/gradients/dense2/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*;
_class1
/-loc:@loss/gradients/dense2/MatMul_grad/MatMul
Т
<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense2/MatMul_grad/MatMul_13^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense2/MatMul_grad/MatMul_1*
_output_shapes
:	А

∞
(loss/gradients/dense1/Relu_grad/ReluGradReluGrad:loss/gradients/dense2/MatMul_grad/tuple/control_dependencydense1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
§
.loss/gradients/dense1/BiasAdd_grad/BiasAddGradBiasAddGrad(loss/gradients/dense1/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:А*
T0
Ч
3loss/gradients/dense1/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense1/BiasAdd_grad/BiasAddGrad)^loss/gradients/dense1/Relu_grad/ReluGrad
Ч
;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/Relu_grad/ReluGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ш
=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense1/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
№
(loss/gradients/dense1/MatMul_grad/MatMulMatMul;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencydense1/kernel/read*
T0*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(
—
*loss/gradients/dense1/MatMul_grad/MatMul_1MatMulPlaceholder_1;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( 
Т
2loss/gradients/dense1/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense1/MatMul_grad/MatMul+^loss/gradients/dense1/MatMul_grad/MatMul_1
Х
:loss/gradients/dense1/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/MatMul_grad/MatMul3^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
У
<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense1/MatMul_grad/MatMul_13^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense1/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
Г
loss/beta1_power/initial_valueConst*
_class
loc:@dense1/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ф
loss/beta1_power
VariableV2*
_class
loc:@dense1/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
љ
loss/beta1_power/AssignAssignloss/beta1_powerloss/beta1_power/initial_value*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
t
loss/beta1_power/readIdentityloss/beta1_power*
_output_shapes
: *
T0*
_class
loc:@dense1/bias
Г
loss/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@dense1/bias*
valueB
 *wЊ?
Ф
loss/beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@dense1/bias*
	container *
shape: 
љ
loss/beta2_power/AssignAssignloss/beta2_powerloss/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
t
loss/beta2_power/readIdentityloss/beta2_power*
_output_shapes
: *
T0*
_class
loc:@dense1/bias
І
4dense1/kernel/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:
С
*dense1/kernel/Adam/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
н
$dense1/kernel/Adam/Initializer/zerosFill4dense1/kernel/Adam/Initializer/zeros/shape_as_tensor*dense1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
РА*
T0* 
_class
loc:@dense1/kernel*

index_type0
ђ
dense1/kernel/Adam
VariableV2*
shared_name * 
_class
loc:@dense1/kernel*
	container *
shape:
РА*
dtype0* 
_output_shapes
:
РА
”
dense1/kernel/Adam/AssignAssigndense1/kernel/Adam$dense1/kernel/Adam/Initializer/zeros*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
Д
dense1/kernel/Adam/readIdentitydense1/kernel/Adam*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
©
6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:
У
,dense1/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense1/kernel*
valueB
 *    
у
&dense1/kernel/Adam_1/Initializer/zerosFill6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense1/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@dense1/kernel*

index_type0* 
_output_shapes
:
РА
Ѓ
dense1/kernel/Adam_1
VariableV2*
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name * 
_class
loc:@dense1/kernel*
	container 
ў
dense1/kernel/Adam_1/AssignAssigndense1/kernel/Adam_1&dense1/kernel/Adam_1/Initializer/zeros*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
И
dense1/kernel/Adam_1/readIdentitydense1/kernel/Adam_1*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
С
"dense1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ю
dense1/bias/Adam
VariableV2*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@dense1/bias
∆
dense1/bias/Adam/AssignAssigndense1/bias/Adam"dense1/bias/Adam/Initializer/zeros*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
y
dense1/bias/Adam/readIdentitydense1/bias/Adam*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
У
$dense1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
†
dense1/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
ћ
dense1/bias/Adam_1/AssignAssigndense1/bias/Adam_1$dense1/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@dense1/bias
}
dense1/bias/Adam_1/readIdentitydense1/bias/Adam_1*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
І
4dense2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense2/kernel*
valueB"   
   
С
*dense2/kernel/Adam/Initializer/zeros/ConstConst* 
_class
loc:@dense2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
м
$dense2/kernel/Adam/Initializer/zerosFill4dense2/kernel/Adam/Initializer/zeros/shape_as_tensor*dense2/kernel/Adam/Initializer/zeros/Const*
T0* 
_class
loc:@dense2/kernel*

index_type0*
_output_shapes
:	А

™
dense2/kernel/Adam
VariableV2*
dtype0*
_output_shapes
:	А
*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	А

“
dense2/kernel/Adam/AssignAssigndense2/kernel/Adam$dense2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	А
*
use_locking(*
T0* 
_class
loc:@dense2/kernel
Г
dense2/kernel/Adam/readIdentitydense2/kernel/Adam*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

©
6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
У
,dense2/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
т
&dense2/kernel/Adam_1/Initializer/zerosFill6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense2/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@dense2/kernel*

index_type0*
_output_shapes
:	А

ђ
dense2/kernel/Adam_1
VariableV2*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	А
*
dtype0*
_output_shapes
:	А

Ў
dense2/kernel/Adam_1/AssignAssigndense2/kernel/Adam_1&dense2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

З
dense2/kernel/Adam_1/readIdentitydense2/kernel/Adam_1*
_output_shapes
:	А
*
T0* 
_class
loc:@dense2/kernel
П
"dense2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
loc:@dense2/bias*
valueB
*    
Ь
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

≈
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
dense2/bias/Adam/readIdentitydense2/bias/Adam*
T0*
_class
loc:@dense2/bias*
_output_shapes
:

С
$dense2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
loc:@dense2/bias*
valueB
*    
Ю
dense2/bias/Adam_1
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

Ћ
dense2/bias/Adam_1/AssignAssigndense2/bias/Adam_1$dense2/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
|
dense2/bias/Adam_1/readIdentitydense2/bias/Adam_1*
T0*
_class
loc:@dense2/bias*
_output_shapes
:

\
loss/Adam/learning_rateConst*
valueB
 *oГ:*
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
 *wЊ?*
dtype0*
_output_shapes
: 
V
loss/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wћ+2
Ь
(loss/Adam/update_dense1/kernel/ApplyAdam	ApplyAdamdense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1*
T0* 
_class
loc:@dense1/kernel*
use_nesterov( * 
_output_shapes
:
РА*
use_locking( 
О
&loss/Adam/update_dense1/bias/ApplyAdam	ApplyAdamdense1/biasdense1/bias/Adamdense1/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
use_nesterov( *
_output_shapes	
:А
Ы
(loss/Adam/update_dense2/kernel/ApplyAdam	ApplyAdamdense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1*
T0* 
_class
loc:@dense2/kernel*
use_nesterov( *
_output_shapes
:	А
*
use_locking( 
Н
&loss/Adam/update_dense2/bias/ApplyAdam	ApplyAdamdense2/biasdense2/bias/Adamdense2/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:
*
use_locking( *
T0*
_class
loc:@dense2/bias
•
loss/Adam/mulMulloss/beta1_power/readloss/Adam/beta1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
•
loss/Adam/AssignAssignloss/beta1_powerloss/Adam/mul*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
І
loss/Adam/mul_1Mulloss/beta2_power/readloss/Adam/beta2'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
©
loss/Adam/Assign_1Assignloss/beta2_powerloss/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@dense1/bias
и
loss/Adam/updateNoOp^loss/Adam/Assign^loss/Adam/Assign_1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam
Р
loss/Adam/valueConst^loss/Adam/update**
_class 
loc:@global_step/global_step*
value	B :*
dtype0*
_output_shapes
: 
†
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
Й
loss/ArgMaxArgMaxdense2/BiasAddloss/ArgMax/dimension*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0
Y
loss/ArgMax_1/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
М
loss/ArgMax_1ArgMaxPlaceholder_2loss/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€
]

loss/EqualEqualloss/ArgMaxloss/ArgMax_1*#
_output_shapes
:€€€€€€€€€*
T0	
Z
	loss/CastCast
loss/Equal*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
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
М
save/SaveV2/tensor_namesConst*њ
valueµB≤Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power*
dtype0*
_output_shapes
:
Г
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
£
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
Ю
save/RestoreV2/tensor_namesConst"/device:CPU:0*њ
valueµB≤Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power*
dtype0*
_output_shapes
:
Х
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
к
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2
ђ
save/AssignAssigncur_epoch/cur_epochsave/RestoreV2*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
•
save/Assign_1Assigndense1/biassave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А
™
save/Assign_2Assigndense1/bias/Adamsave/RestoreV2:2*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ђ
save/Assign_3Assigndense1/bias/Adam_1save/RestoreV2:3*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@dense1/bias
Ѓ
save/Assign_4Assigndense1/kernelsave/RestoreV2:4*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА*
use_locking(
≥
save/Assign_5Assigndense1/kernel/Adamsave/RestoreV2:5*
validate_shape(* 
_output_shapes
:
РА*
use_locking(*
T0* 
_class
loc:@dense1/kernel
µ
save/Assign_6Assigndense1/kernel/Adam_1save/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА
§
save/Assign_7Assigndense2/biassave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

©
save/Assign_8Assigndense2/bias/Adamsave/RestoreV2:8*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
Ђ
save/Assign_9Assigndense2/bias/Adam_1save/RestoreV2:9*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
ѓ
save/Assign_10Assigndense2/kernelsave/RestoreV2:10*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

і
save/Assign_11Assigndense2/kernel/Adamsave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

ґ
save/Assign_12Assigndense2/kernel/Adam_1save/RestoreV2:12*
validate_shape(*
_output_shapes
:	А
*
use_locking(*
T0* 
_class
loc:@dense2/kernel
Ї
save/Assign_13Assignglobal_step/global_stepsave/RestoreV2:13*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: *
use_locking(
І
save/Assign_14Assignloss/beta1_powersave/RestoreV2:14*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense1/bias
І
save/Assign_15Assignloss/beta2_powersave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
Ь
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"AbGує      Пп[и	ю{uњ!„AJжу
еќ
:
Add
x"T
y"T
z"T"
Ttype:
2	
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ы
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
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
s
	AssignAdd
ref"TА

value"T

output_ref"TА" 
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
Р
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

2	Р
Н
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
2	Р
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
Н
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
2	И
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.10.02
b'unknown'И≈
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
ё
global_step/global_step/AssignAssignglobal_step/global_step%global_step/global_step/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0**
_class 
loc:@global_step/global_step
О
global_step/global_step/readIdentityglobal_step/global_step*
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: 
c
!cur_epoch/cur_epoch/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
w
cur_epoch/cur_epoch
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
ќ
cur_epoch/cur_epoch/AssignAssigncur_epoch/cur_epoch!cur_epoch/cur_epoch/initial_value*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
В
cur_epoch/cur_epoch/readIdentitycur_epoch/cur_epoch*
_output_shapes
: *
T0*&
_class
loc:@cur_epoch/cur_epoch
Q
cur_epoch/add/yConst*
dtype0*
_output_shapes
: *
value	B :
`
cur_epoch/addAddcur_epoch/cur_epoch/readcur_epoch/add/y*
_output_shapes
: *
T0
∞
cur_epoch/AssignAssigncur_epoch/cur_epochcur_epoch/add*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
P
PlaceholderPlaceholder*
shape:*
dtype0
*
_output_shapes
:
r
Placeholder_1Placeholder*
dtype0*(
_output_shapes
:€€€€€€€€€Р*
shape:€€€€€€€€€Р
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:€€€€€€€€€
*
shape:€€€€€€€€€

°
.dense1/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:
У
,dense1/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense1/kernel*
valueB
 *HYЛљ*
dtype0*
_output_shapes
: 
У
,dense1/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense1/kernel*
valueB
 *HYЛ=
к
6dense1/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense1/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
РА*

seed *
T0* 
_class
loc:@dense1/kernel*
seed2 
“
,dense1/kernel/Initializer/random_uniform/subSub,dense1/kernel/Initializer/random_uniform/max,dense1/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense1/kernel*
_output_shapes
: 
ж
,dense1/kernel/Initializer/random_uniform/mulMul6dense1/kernel/Initializer/random_uniform/RandomUniform,dense1/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
Ў
(dense1/kernel/Initializer/random_uniformAdd,dense1/kernel/Initializer/random_uniform/mul,dense1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
РА*
T0* 
_class
loc:@dense1/kernel
І
dense1/kernel
VariableV2*
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name * 
_class
loc:@dense1/kernel*
	container 
Ќ
dense1/kernel/AssignAssigndense1/kernel(dense1/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА
z
dense1/kernel/readIdentitydense1/kernel* 
_output_shapes
:
РА*
T0* 
_class
loc:@dense1/kernel
М
dense1/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
_class
loc:@dense1/bias*
valueBА*    
Щ
dense1/bias
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Ј
dense1/bias/AssignAssigndense1/biasdense1/bias/Initializer/zeros*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
o
dense1/bias/readIdentitydense1/bias*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
У
dense1/MatMulMatMulPlaceholder_1dense1/kernel/read*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
transpose_b( *
T0
Д
dense1/BiasAddBiasAdddense1/MatMuldense1/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
V
dense1/ReluReludense1/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
°
.dense2/kernel/Initializer/random_uniform/shapeConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
У
,dense2/kernel/Initializer/random_uniform/minConst* 
_class
loc:@dense2/kernel*
valueB
 *УСџљ*
dtype0*
_output_shapes
: 
У
,dense2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense2/kernel*
valueB
 *УСџ=
й
6dense2/kernel/Initializer/random_uniform/RandomUniformRandomUniform.dense2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А
*

seed *
T0* 
_class
loc:@dense2/kernel*
seed2 
“
,dense2/kernel/Initializer/random_uniform/subSub,dense2/kernel/Initializer/random_uniform/max,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
: 
е
,dense2/kernel/Initializer/random_uniform/mulMul6dense2/kernel/Initializer/random_uniform/RandomUniform,dense2/kernel/Initializer/random_uniform/sub*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

„
(dense2/kernel/Initializer/random_uniformAdd,dense2/kernel/Initializer/random_uniform/mul,dense2/kernel/Initializer/random_uniform/min*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

•
dense2/kernel
VariableV2*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	А
*
dtype0*
_output_shapes
:	А

ћ
dense2/kernel/AssignAssigndense2/kernel(dense2/kernel/Initializer/random_uniform*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

y
dense2/kernel/readIdentitydense2/kernel*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

К
dense2/bias/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:

Ч
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

ґ
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

Р
dense2/MatMulMatMuldense1/Reludense2/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
transpose_b( 
Г
dense2/BiasAddBiasAdddense2/MatMuldense2/bias/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
*
T0
П
>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientPlaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€

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
1loss/softmax_cross_entropy_with_logits_sg/Shape_1Shapedense2/BiasAdd*
_output_shapes
:*
T0*
out_type0
q
/loss/softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
Є
-loss/softmax_cross_entropy_with_logits_sg/SubSub0loss/softmax_cross_entropy_with_logits_sg/Rank_1/loss/softmax_cross_entropy_with_logits_sg/Sub/y*
_output_shapes
: *
T0
¶
5loss/softmax_cross_entropy_with_logits_sg/Slice/beginPack-loss/softmax_cross_entropy_with_logits_sg/Sub*
N*
_output_shapes
:*
T0*

axis 
~
4loss/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
К
/loss/softmax_cross_entropy_with_logits_sg/SliceSlice1loss/softmax_cross_entropy_with_logits_sg/Shape_15loss/softmax_cross_entropy_with_logits_sg/Slice/begin4loss/softmax_cross_entropy_with_logits_sg/Slice/size*
_output_shapes
:*
Index0*
T0
М
9loss/softmax_cross_entropy_with_logits_sg/concat/values_0Const*
dtype0*
_output_shapes
:*
valueB:
€€€€€€€€€
w
5loss/softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Щ
0loss/softmax_cross_entropy_with_logits_sg/concatConcatV29loss/softmax_cross_entropy_with_logits_sg/concat/values_0/loss/softmax_cross_entropy_with_logits_sg/Slice5loss/softmax_cross_entropy_with_logits_sg/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
«
1loss/softmax_cross_entropy_with_logits_sg/ReshapeReshapedense2/BiasAdd0loss/softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
r
0loss/softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
ѓ
1loss/softmax_cross_entropy_with_logits_sg/Shape_2Shape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0*
_output_shapes
:
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
dtype0*
_output_shapes
: *
value	B :
Љ
/loss/softmax_cross_entropy_with_logits_sg/Sub_1Sub0loss/softmax_cross_entropy_with_logits_sg/Rank_21loss/softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0*
_output_shapes
: 
™
7loss/softmax_cross_entropy_with_logits_sg/Slice_1/beginPack/loss/softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N*
_output_shapes
:
А
6loss/softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Р
1loss/softmax_cross_entropy_with_logits_sg/Slice_1Slice1loss/softmax_cross_entropy_with_logits_sg/Shape_27loss/softmax_cross_entropy_with_logits_sg/Slice_1/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_1/size*
_output_shapes
:*
Index0*
T0
О
;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
y
7loss/softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
°
2loss/softmax_cross_entropy_with_logits_sg/concat_1ConcatV2;loss/softmax_cross_entropy_with_logits_sg/concat_1/values_01loss/softmax_cross_entropy_with_logits_sg/Slice_17loss/softmax_cross_entropy_with_logits_sg/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
ы
3loss/softmax_cross_entropy_with_logits_sg/Reshape_1Reshape>loss/softmax_cross_entropy_with_logits_sg/labels_stop_gradient2loss/softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ь
)loss/softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits1loss/softmax_cross_entropy_with_logits_sg/Reshape3loss/softmax_cross_entropy_with_logits_sg/Reshape_1*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€*
T0
s
1loss/softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
Ї
/loss/softmax_cross_entropy_with_logits_sg/Sub_2Sub.loss/softmax_cross_entropy_with_logits_sg/Rank1loss/softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0*
_output_shapes
: 
Б
7loss/softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 
©
6loss/softmax_cross_entropy_with_logits_sg/Slice_2/sizePack/loss/softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N*
_output_shapes
:
О
1loss/softmax_cross_entropy_with_logits_sg/Slice_2Slice/loss/softmax_cross_entropy_with_logits_sg/Shape7loss/softmax_cross_entropy_with_logits_sg/Slice_2/begin6loss/softmax_cross_entropy_with_logits_sg/Slice_2/size*
_output_shapes
:*
Index0*
T0
Ў
3loss/softmax_cross_entropy_with_logits_sg/Reshape_2Reshape)loss/softmax_cross_entropy_with_logits_sg1loss/softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
T

loss/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Р
	loss/MeanMean3loss/softmax_cross_entropy_with_logits_sg/Reshape_2
loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
W
loss/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
]
loss/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
~
loss/gradients/FillFillloss/gradients/Shapeloss/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
u
+loss/gradients/loss/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
•
%loss/gradients/loss/Mean_grad/ReshapeReshapeloss/gradients/Fill+loss/gradients/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Ц
#loss/gradients/loss/Mean_grad/ShapeShape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0*
_output_shapes
:
ґ
"loss/gradients/loss/Mean_grad/TileTile%loss/gradients/loss/Mean_grad/Reshape#loss/gradients/loss/Mean_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€*

Tmultiples0
Ш
%loss/gradients/loss/Mean_grad/Shape_1Shape3loss/softmax_cross_entropy_with_logits_sg/Reshape_2*
_output_shapes
:*
T0*
out_type0
h
%loss/gradients/loss/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
m
#loss/gradients/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
і
"loss/gradients/loss/Mean_grad/ProdProd%loss/gradients/loss/Mean_grad/Shape_1#loss/gradients/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
o
%loss/gradients/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Є
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
†
%loss/gradients/loss/Mean_grad/MaximumMaximum$loss/gradients/loss/Mean_grad/Prod_1'loss/gradients/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
Ю
&loss/gradients/loss/Mean_grad/floordivFloorDiv"loss/gradients/loss/Mean_grad/Prod%loss/gradients/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
В
"loss/gradients/loss/Mean_grad/CastCast&loss/gradients/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
¶
%loss/gradients/loss/Mean_grad/truedivRealDiv"loss/gradients/loss/Mean_grad/Tile"loss/gradients/loss/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
ґ
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape)loss/softmax_cross_entropy_with_logits_sg*
_output_shapes
:*
T0*
out_type0
М
Oloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshape%loss/gradients/loss/Mean_grad/truedivMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
О
loss/gradients/zeros_like	ZerosLike+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ч
Lloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
≥
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeLloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
ъ
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulMulHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims+loss/softmax_cross_entropy_with_logits_sg:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
ƒ
Hloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax1loss/softmax_cross_entropy_with_logits_sg/Reshape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Ќ
Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/NegNegHloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
Щ
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ј
Jloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsOloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeNloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:€€€€€€€€€
Ф
Closs/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1MulJloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1Aloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
а
Nloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOpB^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulD^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1
З
Vloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentityAloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mulO^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*T
_classJ
HFloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul
Н
Xloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1IdentityCloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1O^loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*V
_classL
JHloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/mul_1*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Щ
Kloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapedense2/BiasAdd*
T0*
out_type0*
_output_shapes
:
љ
Mloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeVloss/gradients/loss/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyKloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

»
.loss/gradients/dense2/BiasAdd_grad/BiasAddGradBiasAddGradMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:

Љ
3loss/gradients/dense2/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense2/BiasAdd_grad/BiasAddGradN^loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
а
;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencyIdentityMloss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*
T0*`
_classV
TRloc:@loss/gradients/loss/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape
Ч
=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense2/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:
*
T0*A
_class7
53loc:@loss/gradients/dense2/BiasAdd_grad/BiasAddGrad
№
(loss/gradients/dense2/MatMul_grad/MatMulMatMul;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependencydense2/kernel/read*
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_a( 
ќ
*loss/gradients/dense2/MatMul_grad/MatMul_1MatMuldense1/Relu;loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А
*
transpose_a(*
transpose_b( *
T0
Т
2loss/gradients/dense2/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense2/MatMul_grad/MatMul+^loss/gradients/dense2/MatMul_grad/MatMul_1
Х
:loss/gradients/dense2/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense2/MatMul_grad/MatMul3^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense2/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Т
<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense2/MatMul_grad/MatMul_13^loss/gradients/dense2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense2/MatMul_grad/MatMul_1*
_output_shapes
:	А

∞
(loss/gradients/dense1/Relu_grad/ReluGradReluGrad:loss/gradients/dense2/MatMul_grad/tuple/control_dependencydense1/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
§
.loss/gradients/dense1/BiasAdd_grad/BiasAddGradBiasAddGrad(loss/gradients/dense1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ч
3loss/gradients/dense1/BiasAdd_grad/tuple/group_depsNoOp/^loss/gradients/dense1/BiasAdd_grad/BiasAddGrad)^loss/gradients/dense1/Relu_grad/ReluGrad
Ч
;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/Relu_grad/ReluGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
Ш
=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1Identity.loss/gradients/dense1/BiasAdd_grad/BiasAddGrad4^loss/gradients/dense1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@loss/gradients/dense1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
№
(loss/gradients/dense1/MatMul_grad/MatMulMatMul;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependencydense1/kernel/read*(
_output_shapes
:€€€€€€€€€Р*
transpose_a( *
transpose_b(*
T0
—
*loss/gradients/dense1/MatMul_grad/MatMul_1MatMulPlaceholder_1;loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
РА*
transpose_a(*
transpose_b( 
Т
2loss/gradients/dense1/MatMul_grad/tuple/group_depsNoOp)^loss/gradients/dense1/MatMul_grad/MatMul+^loss/gradients/dense1/MatMul_grad/MatMul_1
Х
:loss/gradients/dense1/MatMul_grad/tuple/control_dependencyIdentity(loss/gradients/dense1/MatMul_grad/MatMul3^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@loss/gradients/dense1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€Р
У
<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1Identity*loss/gradients/dense1/MatMul_grad/MatMul_13^loss/gradients/dense1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@loss/gradients/dense1/MatMul_grad/MatMul_1* 
_output_shapes
:
РА
Г
loss/beta1_power/initial_valueConst*
_class
loc:@dense1/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ф
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
љ
loss/beta1_power/AssignAssignloss/beta1_powerloss/beta1_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense1/bias
t
loss/beta1_power/readIdentityloss/beta1_power*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
Г
loss/beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
_class
loc:@dense1/bias*
valueB
 *wЊ?
Ф
loss/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@dense1/bias*
	container 
љ
loss/beta2_power/AssignAssignloss/beta2_powerloss/beta2_power/initial_value*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
t
loss/beta2_power/readIdentityloss/beta2_power*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
І
4dense1/kernel/Adam/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:
С
*dense1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense1/kernel*
valueB
 *    
н
$dense1/kernel/Adam/Initializer/zerosFill4dense1/kernel/Adam/Initializer/zeros/shape_as_tensor*dense1/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
РА*
T0* 
_class
loc:@dense1/kernel*

index_type0
ђ
dense1/kernel/Adam
VariableV2*
	container *
shape:
РА*
dtype0* 
_output_shapes
:
РА*
shared_name * 
_class
loc:@dense1/kernel
”
dense1/kernel/Adam/AssignAssigndense1/kernel/Adam$dense1/kernel/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА
Д
dense1/kernel/Adam/readIdentitydense1/kernel/Adam* 
_output_shapes
:
РА*
T0* 
_class
loc:@dense1/kernel
©
6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense1/kernel*
valueB"     *
dtype0*
_output_shapes
:
У
,dense1/kernel/Adam_1/Initializer/zeros/ConstConst* 
_class
loc:@dense1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
у
&dense1/kernel/Adam_1/Initializer/zerosFill6dense1/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense1/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_class
loc:@dense1/kernel*

index_type0* 
_output_shapes
:
РА
Ѓ
dense1/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
РА*
shared_name * 
_class
loc:@dense1/kernel*
	container *
shape:
РА
ў
dense1/kernel/Adam_1/AssignAssigndense1/kernel/Adam_1&dense1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense1/kernel*
validate_shape(* 
_output_shapes
:
РА
И
dense1/kernel/Adam_1/readIdentitydense1/kernel/Adam_1*
T0* 
_class
loc:@dense1/kernel* 
_output_shapes
:
РА
С
"dense1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
Ю
dense1/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense1/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
∆
dense1/bias/Adam/AssignAssigndense1/bias/Adam"dense1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А
y
dense1/bias/Adam/readIdentitydense1/bias/Adam*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
У
$dense1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:А*
_class
loc:@dense1/bias*
valueBА*    
†
dense1/bias/Adam_1
VariableV2*
shape:А*
dtype0*
_output_shapes	
:А*
shared_name *
_class
loc:@dense1/bias*
	container 
ћ
dense1/bias/Adam_1/AssignAssigndense1/bias/Adam_1$dense1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А
}
dense1/bias/Adam_1/readIdentitydense1/bias/Adam_1*
T0*
_class
loc:@dense1/bias*
_output_shapes	
:А
І
4dense2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:* 
_class
loc:@dense2/kernel*
valueB"   
   
С
*dense2/kernel/Adam/Initializer/zeros/ConstConst* 
_class
loc:@dense2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
м
$dense2/kernel/Adam/Initializer/zerosFill4dense2/kernel/Adam/Initializer/zeros/shape_as_tensor*dense2/kernel/Adam/Initializer/zeros/Const*
T0* 
_class
loc:@dense2/kernel*

index_type0*
_output_shapes
:	А

™
dense2/kernel/Adam
VariableV2*
shared_name * 
_class
loc:@dense2/kernel*
	container *
shape:	А
*
dtype0*
_output_shapes
:	А

“
dense2/kernel/Adam/AssignAssigndense2/kernel/Adam$dense2/kernel/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

Г
dense2/kernel/Adam/readIdentitydense2/kernel/Adam*
T0* 
_class
loc:@dense2/kernel*
_output_shapes
:	А

©
6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst* 
_class
loc:@dense2/kernel*
valueB"   
   *
dtype0*
_output_shapes
:
У
,dense2/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: * 
_class
loc:@dense2/kernel*
valueB
 *    
т
&dense2/kernel/Adam_1/Initializer/zerosFill6dense2/kernel/Adam_1/Initializer/zeros/shape_as_tensor,dense2/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	А
*
T0* 
_class
loc:@dense2/kernel*

index_type0
ђ
dense2/kernel/Adam_1
VariableV2*
shape:	А
*
dtype0*
_output_shapes
:	А
*
shared_name * 
_class
loc:@dense2/kernel*
	container 
Ў
dense2/kernel/Adam_1/AssignAssigndense2/kernel/Adam_1&dense2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

З
dense2/kernel/Adam_1/readIdentitydense2/kernel/Adam_1*
_output_shapes
:	А
*
T0* 
_class
loc:@dense2/kernel
П
"dense2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
loc:@dense2/bias*
valueB
*    
Ь
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

≈
dense2/bias/Adam/AssignAssigndense2/bias/Adam"dense2/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense2/bias
x
dense2/bias/Adam/readIdentitydense2/bias/Adam*
_output_shapes
:
*
T0*
_class
loc:@dense2/bias
С
$dense2/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense2/bias*
valueB
*    *
dtype0*
_output_shapes
:

Ю
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
Ћ
dense2/bias/Adam_1/AssignAssigndense2/bias/Adam_1$dense2/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
|
dense2/bias/Adam_1/readIdentitydense2/bias/Adam_1*
T0*
_class
loc:@dense2/bias*
_output_shapes
:

\
loss/Adam/learning_rateConst*
valueB
 *oГ:*
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
 *wЊ?*
dtype0*
_output_shapes
: 
V
loss/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Ь
(loss/Adam/update_dense1/kernel/ApplyAdam	ApplyAdamdense1/kerneldense1/kernel/Adamdense1/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( * 
_output_shapes
:
РА*
use_locking( *
T0* 
_class
loc:@dense1/kernel
О
&loss/Adam/update_dense1/bias/ApplyAdam	ApplyAdamdense1/biasdense1/bias/Adamdense1/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
use_nesterov( *
_output_shapes	
:А
Ы
(loss/Adam/update_dense2/kernel/ApplyAdam	ApplyAdamdense2/kerneldense2/kernel/Adamdense2/kernel/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon<loss/gradients/dense2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@dense2/kernel*
use_nesterov( *
_output_shapes
:	А

Н
&loss/Adam/update_dense2/bias/ApplyAdam	ApplyAdamdense2/biasdense2/bias/Adamdense2/bias/Adam_1loss/beta1_power/readloss/beta2_power/readloss/Adam/learning_rateloss/Adam/beta1loss/Adam/beta2loss/Adam/epsilon=loss/gradients/dense2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense2/bias*
use_nesterov( *
_output_shapes
:
*
use_locking( 
•
loss/Adam/mulMulloss/beta1_power/readloss/Adam/beta1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
•
loss/Adam/AssignAssignloss/beta1_powerloss/Adam/mul*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
І
loss/Adam/mul_1Mulloss/beta2_power/readloss/Adam/beta2'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam*
T0*
_class
loc:@dense1/bias*
_output_shapes
: 
©
loss/Adam/Assign_1Assignloss/beta2_powerloss/Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
и
loss/Adam/updateNoOp^loss/Adam/Assign^loss/Adam/Assign_1'^loss/Adam/update_dense1/bias/ApplyAdam)^loss/Adam/update_dense1/kernel/ApplyAdam'^loss/Adam/update_dense2/bias/ApplyAdam)^loss/Adam/update_dense2/kernel/ApplyAdam
Р
loss/Adam/valueConst^loss/Adam/update*
dtype0*
_output_shapes
: **
_class 
loc:@global_step/global_step*
value	B :
†
	loss/Adam	AssignAddglobal_step/global_steploss/Adam/value*
use_locking( *
T0**
_class 
loc:@global_step/global_step*
_output_shapes
: 
W
loss/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
Й
loss/ArgMaxArgMaxdense2/BiasAddloss/ArgMax/dimension*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0*
T0
Y
loss/ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
М
loss/ArgMax_1ArgMaxPlaceholder_2loss/ArgMax_1/dimension*
T0*
output_type0	*#
_output_shapes
:€€€€€€€€€*

Tidx0
]

loss/EqualEqualloss/ArgMaxloss/ArgMax_1*
T0	*#
_output_shapes
:€€€€€€€€€
Z
	loss/CastCast
loss/Equal*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

V
loss/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
j
loss/Mean_1Mean	loss/Castloss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
М
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*њ
valueµB≤Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power
Г
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
£
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
Ю
save/RestoreV2/tensor_namesConst"/device:CPU:0*њ
valueµB≤Bcur_epoch/cur_epochBdense1/biasBdense1/bias/AdamBdense1/bias/Adam_1Bdense1/kernelBdense1/kernel/AdamBdense1/kernel/Adam_1Bdense2/biasBdense2/bias/AdamBdense2/bias/Adam_1Bdense2/kernelBdense2/kernel/AdamBdense2/kernel/Adam_1Bglobal_step/global_stepBloss/beta1_powerBloss/beta2_power*
dtype0*
_output_shapes
:
Х
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
к
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*T
_output_shapesB
@::::::::::::::::
ђ
save/AssignAssigncur_epoch/cur_epochsave/RestoreV2*
use_locking(*
T0*&
_class
loc:@cur_epoch/cur_epoch*
validate_shape(*
_output_shapes
: 
•
save/Assign_1Assigndense1/biassave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А
™
save/Assign_2Assigndense1/bias/Adamsave/RestoreV2:2*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*
_class
loc:@dense1/bias
ђ
save/Assign_3Assigndense1/bias/Adam_1save/RestoreV2:3*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
Ѓ
save/Assign_4Assigndense1/kernelsave/RestoreV2:4*
validate_shape(* 
_output_shapes
:
РА*
use_locking(*
T0* 
_class
loc:@dense1/kernel
≥
save/Assign_5Assigndense1/kernel/Adamsave/RestoreV2:5*
validate_shape(* 
_output_shapes
:
РА*
use_locking(*
T0* 
_class
loc:@dense1/kernel
µ
save/Assign_6Assigndense1/kernel/Adam_1save/RestoreV2:6*
validate_shape(* 
_output_shapes
:
РА*
use_locking(*
T0* 
_class
loc:@dense1/kernel
§
save/Assign_7Assigndense2/biassave/RestoreV2:7*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
loc:@dense2/bias
©
save/Assign_8Assigndense2/bias/Adamsave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:

Ђ
save/Assign_9Assigndense2/bias/Adam_1save/RestoreV2:9*
T0*
_class
loc:@dense2/bias*
validate_shape(*
_output_shapes
:
*
use_locking(
ѓ
save/Assign_10Assigndense2/kernelsave/RestoreV2:10*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А
*
use_locking(
і
save/Assign_11Assigndense2/kernel/Adamsave/RestoreV2:11*
use_locking(*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А

ґ
save/Assign_12Assigndense2/kernel/Adam_1save/RestoreV2:12*
T0* 
_class
loc:@dense2/kernel*
validate_shape(*
_output_shapes
:	А
*
use_locking(
Ї
save/Assign_13Assignglobal_step/global_stepsave/RestoreV2:13*
use_locking(*
T0**
_class 
loc:@global_step/global_step*
validate_shape(*
_output_shapes
: 
І
save/Assign_14Assignloss/beta1_powersave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
І
save/Assign_15Assignloss/beta2_powersave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@dense1/bias*
validate_shape(*
_output_shapes
: 
Ь
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9""
train_op

	loss/Adam"§
	variablesЦУ
Д
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
dense2/bias/Adam_1:0dense2/bias/Adam_1/Assigndense2/bias/Adam_1/read:02&dense2/bias/Adam_1/Initializer/zeros:0"≠
trainable_variablesХТ
k
dense1/kernel:0dense1/kernel/Assigndense1/kernel/read:02*dense1/kernel/Initializer/random_uniform:08
Z
dense1/bias:0dense1/bias/Assigndense1/bias/read:02dense1/bias/Initializer/zeros:08
k
dense2/kernel:0dense2/kernel/Assigndense2/kernel/read:02*dense2/kernel/Initializer/random_uniform:08
Z
dense2/bias:0dense2/bias/Assigndense2/bias/read:02dense2/bias/Initializer/zeros:08 OЄ       »ЅХ	ІМњ!„Ax*

loss_2ґ÷кCЫ®хЌ       Ю	r®Мњ!„Ax*

acc_1    ќ–