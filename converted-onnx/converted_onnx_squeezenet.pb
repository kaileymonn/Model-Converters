
A
data_0Placeholder*
dtype0*
shape:àà
J
fire3/squeeze1x1_w_0Const*
valueB€*
dtype0
=
fire9/expand3x3_b_0Const*
value	B€*
dtype0
<
fire3/expand3x3_b_0Const*
valueB@*
dtype0
=
fire4/squeeze1x1_b_0Const*
valueB *
dtype0
J
fire5/squeeze1x1_w_0Const*
valueB€ *
dtype0
<
fire2/expand3x3_b_0Const*
valueB@*
dtype0
J
fire6/squeeze1x1_w_0Const*
valueB€0*
dtype0
H
fire3/expand3x3_w_0Const*
valueB@*
dtype0
I
fire7/expand1x1_w_0Const*
valueB0À*
dtype0
=
fire8/expand3x3_b_0Const*
value	B€*
dtype0
=
fire9/squeeze1x1_b_0Const*
valueB@*
dtype0
=
fire2/squeeze1x1_b_0Const*
valueB*
dtype0
I
fire8/expand3x3_w_0Const*
valueB@€*
dtype0
<
fire2/expand1x1_b_0Const*
valueB@*
dtype0
J
fire4/squeeze1x1_w_0Const*
valueB€ *
dtype0
A

conv10_w_0Const*
valueB€è*
dtype0
I
fire9/expand1x1_w_0Const*
valueB@€*
dtype0
I
fire9/expand3x3_w_0Const*
valueB@€*
dtype0
=
fire6/expand1x1_b_0Const*
value	BÀ*
dtype0
4

conv10_b_0Const*
value	Bè*
dtype0
=
fire5/expand1x1_b_0Const*
value	B€*
dtype0
J
fire7/squeeze1x1_w_0Const*
valueB€0*
dtype0
I
fire5/expand3x3_w_0Const*
valueB €*
dtype0
I
fire6/expand3x3_w_0Const*
valueB0À*
dtype0
=
fire5/expand3x3_b_0Const*
value	B€*
dtype0
I
fire4/expand3x3_w_0Const*
valueB €*
dtype0
I
fire6/expand1x1_w_0Const*
valueB0À*
dtype0
=
fire8/squeeze1x1_b_0Const*
valueB@*
dtype0
=
fire8/expand1x1_b_0Const*
value	B€*
dtype0
>
	conv1_w_0Const*
valueB@*
dtype0
I
fire4/expand1x1_w_0Const*
valueB €*
dtype0
=
fire5/squeeze1x1_b_0Const*
valueB *
dtype0
I
fire5/expand1x1_w_0Const*
valueB €*
dtype0
H
fire2/expand3x3_w_0Const*
valueB@*
dtype0
=
fire4/expand3x3_b_0Const*
value	B€*
dtype0
J
fire8/squeeze1x1_w_0Const*
valueB€@*
dtype0
I
fire8/expand1x1_w_0Const*
valueB@€*
dtype0
=
fire9/expand1x1_b_0Const*
value	B€*
dtype0
=
fire7/squeeze1x1_b_0Const*
valueB0*
dtype0
I
fire2/squeeze1x1_w_0Const*
valueB@*
dtype0
=
fire6/expand3x3_b_0Const*
value	BÀ*
dtype0
=
fire7/expand1x1_b_0Const*
value	BÀ*
dtype0
I
fire7/expand3x3_w_0Const*
valueB0À*
dtype0
2
	conv1_b_0Const*
valueB@*
dtype0
=
fire7/expand3x3_b_0Const*
value	BÀ*
dtype0
=
fire6/squeeze1x1_b_0Const*
valueB0*
dtype0
=
fire4/expand1x1_b_0Const*
value	B€*
dtype0
<
fire3/expand1x1_b_0Const*
valueB@*
dtype0
H
fire3/expand1x1_w_0Const*
valueB@*
dtype0
=
fire3/squeeze1x1_b_0Const*
valueB*
dtype0
H
fire2/expand1x1_w_0Const*
valueB@*
dtype0
J
fire9/squeeze1x1_w_0Const*
valueB€@*
dtype0
C
conv1_1/kernelConst*
valueB@*
dtype0
¡
conv1_1Conv2Ddata_0conv1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
!
conv1_2Reluconv1_1*
T0
w
pool1_1MaxPoolconv1_2*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

N
fire2/squeeze1x1_1/kernelConst*
valueB@*
dtype0
¸
fire2/squeeze1x1_1Conv2Dpool1_1fire2/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire2/squeeze1x1_2Relufire2/squeeze1x1_1*
T0
M
fire2/expand1x1_1/kernelConst*
valueB@*
dtype0
Á
fire2/expand1x1_1Conv2Dfire2/squeeze1x1_2fire2/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire2/expand1x1_2Relufire2/expand1x1_1*
T0
M
fire2/expand3x3_1/kernelConst*
valueB@*
dtype0
À
fire2/expand3x3_1Conv2Dfire2/squeeze1x1_2fire2/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire2/expand3x3_2Relufire2/expand3x3_1*
T0
D
fire2/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire2/concat_1ConcatV2fire2/expand1x1_2fire2/expand3x3_2fire2/concat_1/axis*
T0*
N*

Tidx0
O
fire3/squeeze1x1_1/kernelConst*
valueB€*
dtype0
¿
fire3/squeeze1x1_1Conv2Dfire2/concat_1fire3/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire3/squeeze1x1_2Relufire3/squeeze1x1_1*
T0
M
fire3/expand1x1_1/kernelConst*
valueB@*
dtype0
Á
fire3/expand1x1_1Conv2Dfire3/squeeze1x1_2fire3/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire3/expand1x1_2Relufire3/expand1x1_1*
T0
M
fire3/expand3x3_1/kernelConst*
valueB@*
dtype0
À
fire3/expand3x3_1Conv2Dfire3/squeeze1x1_2fire3/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire3/expand3x3_2Relufire3/expand3x3_1*
T0
D
fire3/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire3/concat_1ConcatV2fire3/expand1x1_2fire3/expand3x3_2fire3/concat_1/axis*
T0*
N*

Tidx0
~
pool3_1MaxPoolfire3/concat_1*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

O
fire4/squeeze1x1_1/kernelConst*
valueB€ *
dtype0
¸
fire4/squeeze1x1_1Conv2Dpool3_1fire4/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire4/squeeze1x1_2Relufire4/squeeze1x1_1*
T0
N
fire4/expand1x1_1/kernelConst*
valueB €*
dtype0
Á
fire4/expand1x1_1Conv2Dfire4/squeeze1x1_2fire4/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire4/expand1x1_2Relufire4/expand1x1_1*
T0
N
fire4/expand3x3_1/kernelConst*
valueB €*
dtype0
À
fire4/expand3x3_1Conv2Dfire4/squeeze1x1_2fire4/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire4/expand3x3_2Relufire4/expand3x3_1*
T0
D
fire4/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire4/concat_1ConcatV2fire4/expand1x1_2fire4/expand3x3_2fire4/concat_1/axis*
T0*
N*

Tidx0
O
fire5/squeeze1x1_1/kernelConst*
valueB€ *
dtype0
¿
fire5/squeeze1x1_1Conv2Dfire4/concat_1fire5/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire5/squeeze1x1_2Relufire5/squeeze1x1_1*
T0
N
fire5/expand1x1_1/kernelConst*
valueB €*
dtype0
Á
fire5/expand1x1_1Conv2Dfire5/squeeze1x1_2fire5/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire5/expand1x1_2Relufire5/expand1x1_1*
T0
N
fire5/expand3x3_1/kernelConst*
valueB €*
dtype0
À
fire5/expand3x3_1Conv2Dfire5/squeeze1x1_2fire5/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire5/expand3x3_2Relufire5/expand3x3_1*
T0
D
fire5/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire5/concat_1ConcatV2fire5/expand1x1_2fire5/expand3x3_2fire5/concat_1/axis*
T0*
N*

Tidx0
~
pool5_1MaxPoolfire5/concat_1*
T0*
strides
*
data_formatNHWC*
paddingVALID*
ksize

O
fire6/squeeze1x1_1/kernelConst*
valueB€0*
dtype0
¸
fire6/squeeze1x1_1Conv2Dpool5_1fire6/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire6/squeeze1x1_2Relufire6/squeeze1x1_1*
T0
N
fire6/expand1x1_1/kernelConst*
valueB0À*
dtype0
Á
fire6/expand1x1_1Conv2Dfire6/squeeze1x1_2fire6/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire6/expand1x1_2Relufire6/expand1x1_1*
T0
N
fire6/expand3x3_1/kernelConst*
valueB0À*
dtype0
À
fire6/expand3x3_1Conv2Dfire6/squeeze1x1_2fire6/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire6/expand3x3_2Relufire6/expand3x3_1*
T0
D
fire6/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire6/concat_1ConcatV2fire6/expand1x1_2fire6/expand3x3_2fire6/concat_1/axis*
T0*
N*

Tidx0
O
fire7/squeeze1x1_1/kernelConst*
valueB€0*
dtype0
¿
fire7/squeeze1x1_1Conv2Dfire6/concat_1fire7/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire7/squeeze1x1_2Relufire7/squeeze1x1_1*
T0
N
fire7/expand1x1_1/kernelConst*
valueB0À*
dtype0
Á
fire7/expand1x1_1Conv2Dfire7/squeeze1x1_2fire7/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire7/expand1x1_2Relufire7/expand1x1_1*
T0
N
fire7/expand3x3_1/kernelConst*
valueB0À*
dtype0
À
fire7/expand3x3_1Conv2Dfire7/squeeze1x1_2fire7/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire7/expand3x3_2Relufire7/expand3x3_1*
T0
D
fire7/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire7/concat_1ConcatV2fire7/expand1x1_2fire7/expand3x3_2fire7/concat_1/axis*
T0*
N*

Tidx0
O
fire8/squeeze1x1_1/kernelConst*
valueB€@*
dtype0
¿
fire8/squeeze1x1_1Conv2Dfire7/concat_1fire8/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire8/squeeze1x1_2Relufire8/squeeze1x1_1*
T0
N
fire8/expand1x1_1/kernelConst*
valueB@€*
dtype0
Á
fire8/expand1x1_1Conv2Dfire8/squeeze1x1_2fire8/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire8/expand1x1_2Relufire8/expand1x1_1*
T0
N
fire8/expand3x3_1/kernelConst*
valueB@€*
dtype0
À
fire8/expand3x3_1Conv2Dfire8/squeeze1x1_2fire8/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire8/expand3x3_2Relufire8/expand3x3_1*
T0
D
fire8/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire8/concat_1ConcatV2fire8/expand1x1_2fire8/expand3x3_2fire8/concat_1/axis*
T0*
N*

Tidx0
O
fire9/squeeze1x1_1/kernelConst*
valueB€@*
dtype0
¿
fire9/squeeze1x1_1Conv2Dfire8/concat_1fire9/squeeze1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
7
fire9/squeeze1x1_2Relufire9/squeeze1x1_1*
T0
N
fire9/expand1x1_1/kernelConst*
valueB@€*
dtype0
Á
fire9/expand1x1_1Conv2Dfire9/squeeze1x1_2fire9/expand1x1_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
5
fire9/expand1x1_2Relufire9/expand1x1_1*
T0
N
fire9/expand3x3_1/kernelConst*
valueB@€*
dtype0
À
fire9/expand3x3_1Conv2Dfire9/squeeze1x1_2fire9/expand3x3_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingSAME*
use_cudnn_on_gpu(
5
fire9/expand3x3_2Relufire9/expand3x3_1*
T0
D
fire9/concat_1/axisConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
s
fire9/concat_1ConcatV2fire9/expand1x1_2fire9/expand3x3_2fire9/concat_1/axis*
T0*
N*

Tidx0
3
fire9/concat_2Identityfire9/concat_1*
T0
F
conv10_1/kernelConst*
valueB€è*
dtype0
«
conv10_1Conv2Dfire9/concat_2conv10_1/kernel*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(
#
conv10_2Reluconv10_1*
T0
d
/global_average_pooling2d/Mean/reduction_indicesConst*
valueB"      *
dtype0
†
global_average_pooling2d/MeanMeanconv10_2/global_average_pooling2d/Mean/reduction_indices*
	keep_dims( *
T0*

Tidx0
8
pool10_1_1/dimConst*
value	B :*
dtype0
\

pool10_1_1
ExpandDimsglobal_average_pooling2d/Meanpool10_1_1/dim*
T0*

Tdim0
6
pool10_1/dimConst*
value	B :*
dtype0
E
pool10_1
ExpandDims
pool10_1_1pool10_1/dim*
T0*

Tdim0
*
softmaxout_1Softmaxpool10_1*
T0" 