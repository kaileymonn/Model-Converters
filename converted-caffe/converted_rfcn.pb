
?
dataPlaceholder*
dtype0*
shape:��
A
conv1/kernelConst*
valueB@*
dtype0
�
conv1Conv2Ddataconv1/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
:batch_normalization/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:@*,
_class"
 loc:@batch_normalization/gamma
�
0batch_normalization/gamma/Initializer/ones/ConstConst*
dtype0*
valueB
 *  �?*,
_class"
 loc:@batch_normalization/gamma
�
*batch_normalization/gamma/Initializer/onesFill:batch_normalization/gamma/Initializer/ones/shape_as_tensor0batch_normalization/gamma/Initializer/ones/Const*
T0*

index_type0*,
_class"
 loc:@batch_normalization/gamma
�
batch_normalization/gamma
VariableV2*
shared_name *,
_class"
 loc:@batch_normalization/gamma*
dtype0*
	container *
shape:@
�
 batch_normalization/gamma/AssignAssignbatch_normalization/gamma*batch_normalization/gamma/Initializer/ones*
use_locking(*
T0*,
_class"
 loc:@batch_normalization/gamma*
validate_shape(
|
batch_normalization/gamma/readIdentitybatch_normalization/gamma*
T0*,
_class"
 loc:@batch_normalization/gamma
�
:batch_normalization/beta/Initializer/zeros/shape_as_tensorConst*
valueB:@*+
_class!
loc:@batch_normalization/beta*
dtype0
�
0batch_normalization/beta/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@batch_normalization/beta*
dtype0
�
*batch_normalization/beta/Initializer/zerosFill:batch_normalization/beta/Initializer/zeros/shape_as_tensor0batch_normalization/beta/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@batch_normalization/beta
�
batch_normalization/beta
VariableV2*
	container *
shape:@*
shared_name *+
_class!
loc:@batch_normalization/beta*
dtype0
�
batch_normalization/beta/AssignAssignbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*+
_class!
loc:@batch_normalization/beta*
validate_shape(*
use_locking(*
T0
y
batch_normalization/beta/readIdentitybatch_normalization/beta*
T0*+
_class!
loc:@batch_normalization/beta
�
Abatch_normalization/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:@*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
�
7batch_normalization/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
�
1batch_normalization/moving_mean/Initializer/zerosFillAbatch_normalization/moving_mean/Initializer/zeros/shape_as_tensor7batch_normalization/moving_mean/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@batch_normalization/moving_mean
�
batch_normalization/moving_mean
VariableV2*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
	container *
shape:@*
shared_name 
�
&batch_normalization/moving_mean/AssignAssignbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
validate_shape(*
use_locking(
�
$batch_normalization/moving_mean/readIdentitybatch_normalization/moving_mean*
T0*2
_class(
&$loc:@batch_normalization/moving_mean
�
Dbatch_normalization/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:@*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
�
:batch_normalization/moving_variance/Initializer/ones/ConstConst*
valueB
 *  �?*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
�
4batch_normalization/moving_variance/Initializer/onesFillDbatch_normalization/moving_variance/Initializer/ones/shape_as_tensor:batch_normalization/moving_variance/Initializer/ones/Const*
T0*

index_type0*6
_class,
*(loc:@batch_normalization/moving_variance
�
#batch_normalization/moving_variance
VariableV2*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0*
	container *
shape:@*
shared_name 
�
*batch_normalization/moving_variance/AssignAssign#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*
use_locking(*
T0*6
_class,
*(loc:@batch_normalization/moving_variance*
validate_shape(
�
(batch_normalization/moving_variance/readIdentity#batch_normalization/moving_variance*
T0*6
_class,
*(loc:@batch_normalization/moving_variance
�
"batch_normalization/FusedBatchNormFusedBatchNormconv1batch_normalization/gamma/readbatch_normalization/beta/read$batch_normalization/moving_mean/read(batch_normalization/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
F
batch_normalization/ConstConst*
valueB
 * �:*
dtype0
A
bn_conv1Identity"batch_normalization/FusedBatchNorm*
T0
*
scale_conv1Identitybn_conv1*
T0
(

conv1_reluReluscale_conv1*
T0
w
pool1MaxPool
conv1_relu*
paddingSAME*
T0*
data_formatNHWC*
strides
*
ksize

J
res2a_branch1/kernelConst*
valueB@�*
dtype0
�
res2a_branch1Conv2Dpool1res2a_branch1/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
<batch_normalization/gamma_1/Initializer/ones/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/gamma_1*
valueB:�*
dtype0
�
2batch_normalization/gamma_1/Initializer/ones/ConstConst*.
_class$
" loc:@batch_normalization/gamma_1*
valueB
 *  �?*
dtype0
�
,batch_normalization/gamma_1/Initializer/onesFill<batch_normalization/gamma_1/Initializer/ones/shape_as_tensor2batch_normalization/gamma_1/Initializer/ones/Const*
T0*.
_class$
" loc:@batch_normalization/gamma_1*

index_type0
�
batch_normalization/gamma_1
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/gamma_1
�
"batch_normalization/gamma_1/AssignAssignbatch_normalization/gamma_1,batch_normalization/gamma_1/Initializer/ones*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_1
�
 batch_normalization/gamma_1/readIdentitybatch_normalization/gamma_1*.
_class$
" loc:@batch_normalization/gamma_1*
T0
�
<batch_normalization/beta_1/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@batch_normalization/beta_1*
valueB:�*
dtype0
�
2batch_normalization/beta_1/Initializer/zeros/ConstConst*-
_class#
!loc:@batch_normalization/beta_1*
valueB
 *    *
dtype0
�
,batch_normalization/beta_1/Initializer/zerosFill<batch_normalization/beta_1/Initializer/zeros/shape_as_tensor2batch_normalization/beta_1/Initializer/zeros/Const*-
_class#
!loc:@batch_normalization/beta_1*

index_type0*
T0
�
batch_normalization/beta_1
VariableV2*
shape:�*
shared_name *-
_class#
!loc:@batch_normalization/beta_1*
dtype0*
	container 
�
!batch_normalization/beta_1/AssignAssignbatch_normalization/beta_1,batch_normalization/beta_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_1*
validate_shape(

batch_normalization/beta_1/readIdentitybatch_normalization/beta_1*
T0*-
_class#
!loc:@batch_normalization/beta_1
�
Cbatch_normalization/moving_mean_1/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@batch_normalization/moving_mean_1*
valueB:�*
dtype0
�
9batch_normalization/moving_mean_1/Initializer/zeros/ConstConst*4
_class*
(&loc:@batch_normalization/moving_mean_1*
valueB
 *    *
dtype0
�
3batch_normalization/moving_mean_1/Initializer/zerosFillCbatch_normalization/moving_mean_1/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_1/Initializer/zeros/Const*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_1*

index_type0
�
!batch_normalization/moving_mean_1
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_1*
dtype0*
	container *
shape:�
�
(batch_normalization/moving_mean_1/AssignAssign!batch_normalization/moving_mean_13batch_normalization/moving_mean_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_1*
validate_shape(
�
&batch_normalization/moving_mean_1/readIdentity!batch_normalization/moving_mean_1*4
_class*
(&loc:@batch_normalization/moving_mean_1*
T0
�
Fbatch_normalization/moving_variance_1/Initializer/ones/shape_as_tensorConst*8
_class.
,*loc:@batch_normalization/moving_variance_1*
valueB:�*
dtype0
�
<batch_normalization/moving_variance_1/Initializer/ones/ConstConst*
dtype0*8
_class.
,*loc:@batch_normalization/moving_variance_1*
valueB
 *  �?
�
6batch_normalization/moving_variance_1/Initializer/onesFillFbatch_normalization/moving_variance_1/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_1/Initializer/ones/Const*8
_class.
,*loc:@batch_normalization/moving_variance_1*

index_type0*
T0
�
%batch_normalization/moving_variance_1
VariableV2*8
_class.
,*loc:@batch_normalization/moving_variance_1*
dtype0*
	container *
shape:�*
shared_name 
�
,batch_normalization/moving_variance_1/AssignAssign%batch_normalization/moving_variance_16batch_normalization/moving_variance_1/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_1*
validate_shape(
�
*batch_normalization/moving_variance_1/readIdentity%batch_normalization/moving_variance_1*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_1
�
$batch_normalization/FusedBatchNorm_1FusedBatchNormres2a_branch1 batch_normalization/gamma_1/readbatch_normalization/beta_1/read&batch_normalization/moving_mean_1/read*batch_normalization/moving_variance_1/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
H
batch_normalization/Const_1Const*
valueB
 * �:*
dtype0
E
bn2a_branch1Identity"batch_normalization/FusedBatchNorm*
T0
2
scale2a_branch1Identitybn2a_branch1*
T0
J
res2a_branch2a/kernelConst*
valueB@@*
dtype0
�
res2a_branch2aConv2Dpool1res2a_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
<batch_normalization/gamma_2/Initializer/ones/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization/gamma_2*
dtype0
�
2batch_normalization/gamma_2/Initializer/ones/ConstConst*
valueB
 *  �?*.
_class$
" loc:@batch_normalization/gamma_2*
dtype0
�
,batch_normalization/gamma_2/Initializer/onesFill<batch_normalization/gamma_2/Initializer/ones/shape_as_tensor2batch_normalization/gamma_2/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/gamma_2
�
batch_normalization/gamma_2
VariableV2*
shape:@*
shared_name *.
_class$
" loc:@batch_normalization/gamma_2*
dtype0*
	container 
�
"batch_normalization/gamma_2/AssignAssignbatch_normalization/gamma_2,batch_normalization/gamma_2/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_2*
validate_shape(
�
 batch_normalization/gamma_2/readIdentitybatch_normalization/gamma_2*
T0*.
_class$
" loc:@batch_normalization/gamma_2
�
<batch_normalization/beta_2/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization/beta_2*
dtype0
�
2batch_normalization/beta_2/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization/beta_2*
dtype0
�
,batch_normalization/beta_2/Initializer/zerosFill<batch_normalization/beta_2/Initializer/zeros/shape_as_tensor2batch_normalization/beta_2/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization/beta_2
�
batch_normalization/beta_2
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization/beta_2*
dtype0*
	container *
shape:@
�
!batch_normalization/beta_2/AssignAssignbatch_normalization/beta_2,batch_normalization/beta_2/Initializer/zeros*-
_class#
!loc:@batch_normalization/beta_2*
validate_shape(*
use_locking(*
T0

batch_normalization/beta_2/readIdentitybatch_normalization/beta_2*
T0*-
_class#
!loc:@batch_normalization/beta_2
�
Cbatch_normalization/moving_mean_2/Initializer/zeros/shape_as_tensorConst*
valueB:@*4
_class*
(&loc:@batch_normalization/moving_mean_2*
dtype0
�
9batch_normalization/moving_mean_2/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization/moving_mean_2*
dtype0
�
3batch_normalization/moving_mean_2/Initializer/zerosFillCbatch_normalization/moving_mean_2/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_2/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization/moving_mean_2
�
!batch_normalization/moving_mean_2
VariableV2*4
_class*
(&loc:@batch_normalization/moving_mean_2*
dtype0*
	container *
shape:@*
shared_name 
�
(batch_normalization/moving_mean_2/AssignAssign!batch_normalization/moving_mean_23batch_normalization/moving_mean_2/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_2*
validate_shape(
�
&batch_normalization/moving_mean_2/readIdentity!batch_normalization/moving_mean_2*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_2
�
Fbatch_normalization/moving_variance_2/Initializer/ones/shape_as_tensorConst*
valueB:@*8
_class.
,*loc:@batch_normalization/moving_variance_2*
dtype0
�
<batch_normalization/moving_variance_2/Initializer/ones/ConstConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization/moving_variance_2*
dtype0
�
6batch_normalization/moving_variance_2/Initializer/onesFillFbatch_normalization/moving_variance_2/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_2/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization/moving_variance_2
�
%batch_normalization/moving_variance_2
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_2*
dtype0*
	container *
shape:@
�
,batch_normalization/moving_variance_2/AssignAssign%batch_normalization/moving_variance_26batch_normalization/moving_variance_2/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_2*
validate_shape(
�
*batch_normalization/moving_variance_2/readIdentity%batch_normalization/moving_variance_2*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_2
�
$batch_normalization/FusedBatchNorm_2FusedBatchNormres2a_branch2a batch_normalization/gamma_2/readbatch_normalization/beta_2/read&batch_normalization/moving_mean_2/read*batch_normalization/moving_variance_2/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
H
batch_normalization/Const_2Const*
valueB
 * �:*
dtype0
F
bn2a_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2a_branch2aIdentitybn2a_branch2a*
T0
6
res2a_branch2a_reluReluscale2a_branch2a*
T0
J
res2a_branch2b/kernelConst*
valueB@@*
dtype0
�
res2a_branch2bConv2Dres2a_branch2a_relures2a_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
<batch_normalization/gamma_3/Initializer/ones/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/gamma_3*
valueB:@*
dtype0
�
2batch_normalization/gamma_3/Initializer/ones/ConstConst*.
_class$
" loc:@batch_normalization/gamma_3*
valueB
 *  �?*
dtype0
�
,batch_normalization/gamma_3/Initializer/onesFill<batch_normalization/gamma_3/Initializer/ones/shape_as_tensor2batch_normalization/gamma_3/Initializer/ones/Const*
T0*.
_class$
" loc:@batch_normalization/gamma_3*

index_type0
�
batch_normalization/gamma_3
VariableV2*
shape:@*
shared_name *.
_class$
" loc:@batch_normalization/gamma_3*
dtype0*
	container 
�
"batch_normalization/gamma_3/AssignAssignbatch_normalization/gamma_3,batch_normalization/gamma_3/Initializer/ones*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_3
�
 batch_normalization/gamma_3/readIdentitybatch_normalization/gamma_3*.
_class$
" loc:@batch_normalization/gamma_3*
T0
�
<batch_normalization/beta_3/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@batch_normalization/beta_3*
valueB:@*
dtype0
�
2batch_normalization/beta_3/Initializer/zeros/ConstConst*-
_class#
!loc:@batch_normalization/beta_3*
valueB
 *    *
dtype0
�
,batch_normalization/beta_3/Initializer/zerosFill<batch_normalization/beta_3/Initializer/zeros/shape_as_tensor2batch_normalization/beta_3/Initializer/zeros/Const*-
_class#
!loc:@batch_normalization/beta_3*

index_type0*
T0
�
batch_normalization/beta_3
VariableV2*
shared_name *-
_class#
!loc:@batch_normalization/beta_3*
dtype0*
	container *
shape:@
�
!batch_normalization/beta_3/AssignAssignbatch_normalization/beta_3,batch_normalization/beta_3/Initializer/zeros*
validate_shape(*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_3

batch_normalization/beta_3/readIdentitybatch_normalization/beta_3*
T0*-
_class#
!loc:@batch_normalization/beta_3
�
Cbatch_normalization/moving_mean_3/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@batch_normalization/moving_mean_3*
valueB:@*
dtype0
�
9batch_normalization/moving_mean_3/Initializer/zeros/ConstConst*4
_class*
(&loc:@batch_normalization/moving_mean_3*
valueB
 *    *
dtype0
�
3batch_normalization/moving_mean_3/Initializer/zerosFillCbatch_normalization/moving_mean_3/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_3/Initializer/zeros/Const*4
_class*
(&loc:@batch_normalization/moving_mean_3*

index_type0*
T0
�
!batch_normalization/moving_mean_3
VariableV2*
shape:@*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_3*
dtype0*
	container 
�
(batch_normalization/moving_mean_3/AssignAssign!batch_normalization/moving_mean_33batch_normalization/moving_mean_3/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_3*
validate_shape(
�
&batch_normalization/moving_mean_3/readIdentity!batch_normalization/moving_mean_3*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_3
�
Fbatch_normalization/moving_variance_3/Initializer/ones/shape_as_tensorConst*8
_class.
,*loc:@batch_normalization/moving_variance_3*
valueB:@*
dtype0
�
<batch_normalization/moving_variance_3/Initializer/ones/ConstConst*
dtype0*8
_class.
,*loc:@batch_normalization/moving_variance_3*
valueB
 *  �?
�
6batch_normalization/moving_variance_3/Initializer/onesFillFbatch_normalization/moving_variance_3/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_3/Initializer/ones/Const*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_3*

index_type0
�
%batch_normalization/moving_variance_3
VariableV2*
shape:@*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_3*
dtype0*
	container 
�
,batch_normalization/moving_variance_3/AssignAssign%batch_normalization/moving_variance_36batch_normalization/moving_variance_3/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_3*
validate_shape(
�
*batch_normalization/moving_variance_3/readIdentity%batch_normalization/moving_variance_3*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_3
�
$batch_normalization/FusedBatchNorm_3FusedBatchNormres2a_branch2b batch_normalization/gamma_3/readbatch_normalization/beta_3/read&batch_normalization/moving_mean_3/read*batch_normalization/moving_variance_3/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
H
batch_normalization/Const_3Const*
dtype0*
valueB
 * �:
F
bn2a_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2a_branch2bIdentitybn2a_branch2b*
T0
6
res2a_branch2b_reluReluscale2a_branch2b*
T0
K
res2a_branch2c/kernelConst*
valueB@�*
dtype0
�
res2a_branch2cConv2Dres2a_branch2b_relures2a_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
<batch_normalization/gamma_4/Initializer/ones/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/gamma_4*
dtype0
�
2batch_normalization/gamma_4/Initializer/ones/ConstConst*
valueB
 *  �?*.
_class$
" loc:@batch_normalization/gamma_4*
dtype0
�
,batch_normalization/gamma_4/Initializer/onesFill<batch_normalization/gamma_4/Initializer/ones/shape_as_tensor2batch_normalization/gamma_4/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/gamma_4
�
batch_normalization/gamma_4
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/gamma_4*
dtype0*
	container 
�
"batch_normalization/gamma_4/AssignAssignbatch_normalization/gamma_4,batch_normalization/gamma_4/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_4*
validate_shape(
�
 batch_normalization/gamma_4/readIdentitybatch_normalization/gamma_4*
T0*.
_class$
" loc:@batch_normalization/gamma_4
�
<batch_normalization/beta_4/Initializer/zeros/shape_as_tensorConst*
valueB:�*-
_class#
!loc:@batch_normalization/beta_4*
dtype0
�
2batch_normalization/beta_4/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization/beta_4*
dtype0
�
,batch_normalization/beta_4/Initializer/zerosFill<batch_normalization/beta_4/Initializer/zeros/shape_as_tensor2batch_normalization/beta_4/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization/beta_4
�
batch_normalization/beta_4
VariableV2*
	container *
shape:�*
shared_name *-
_class#
!loc:@batch_normalization/beta_4*
dtype0
�
!batch_normalization/beta_4/AssignAssignbatch_normalization/beta_4,batch_normalization/beta_4/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_4*
validate_shape(

batch_normalization/beta_4/readIdentitybatch_normalization/beta_4*
T0*-
_class#
!loc:@batch_normalization/beta_4
�
Cbatch_normalization/moving_mean_4/Initializer/zeros/shape_as_tensorConst*
valueB:�*4
_class*
(&loc:@batch_normalization/moving_mean_4*
dtype0
�
9batch_normalization/moving_mean_4/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization/moving_mean_4*
dtype0
�
3batch_normalization/moving_mean_4/Initializer/zerosFillCbatch_normalization/moving_mean_4/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_4/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization/moving_mean_4
�
!batch_normalization/moving_mean_4
VariableV2*
shape:�*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_4*
dtype0*
	container 
�
(batch_normalization/moving_mean_4/AssignAssign!batch_normalization/moving_mean_43batch_normalization/moving_mean_4/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_4*
validate_shape(
�
&batch_normalization/moving_mean_4/readIdentity!batch_normalization/moving_mean_4*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_4
�
Fbatch_normalization/moving_variance_4/Initializer/ones/shape_as_tensorConst*
valueB:�*8
_class.
,*loc:@batch_normalization/moving_variance_4*
dtype0
�
<batch_normalization/moving_variance_4/Initializer/ones/ConstConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization/moving_variance_4*
dtype0
�
6batch_normalization/moving_variance_4/Initializer/onesFillFbatch_normalization/moving_variance_4/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_4/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization/moving_variance_4
�
%batch_normalization/moving_variance_4
VariableV2*
	container *
shape:�*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_4*
dtype0
�
,batch_normalization/moving_variance_4/AssignAssign%batch_normalization/moving_variance_46batch_normalization/moving_variance_4/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_4*
validate_shape(
�
*batch_normalization/moving_variance_4/readIdentity%batch_normalization/moving_variance_4*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_4
�
$batch_normalization/FusedBatchNorm_4FusedBatchNormres2a_branch2c batch_normalization/gamma_4/readbatch_normalization/beta_4/read&batch_normalization/moving_mean_4/read*batch_normalization/moving_variance_4/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
H
batch_normalization/Const_4Const*
valueB
 * �:*
dtype0
F
bn2a_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2a_branch2cIdentitybn2a_branch2c*
T0
8
res2aAddscale2a_branch1scale2a_branch2c*
T0
"

res2a_reluRelures2a*
T0
J
res2b_branch2a/kernelConst*
valueB@@*
dtype0
�
res2b_branch2aConv2D
res2a_relures2b_branch2a/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
<batch_normalization/gamma_5/Initializer/ones/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/gamma_5*
valueB:@*
dtype0
�
2batch_normalization/gamma_5/Initializer/ones/ConstConst*.
_class$
" loc:@batch_normalization/gamma_5*
valueB
 *  �?*
dtype0
�
,batch_normalization/gamma_5/Initializer/onesFill<batch_normalization/gamma_5/Initializer/ones/shape_as_tensor2batch_normalization/gamma_5/Initializer/ones/Const*.
_class$
" loc:@batch_normalization/gamma_5*

index_type0*
T0
�
batch_normalization/gamma_5
VariableV2*
	container *
shape:@*
shared_name *.
_class$
" loc:@batch_normalization/gamma_5*
dtype0
�
"batch_normalization/gamma_5/AssignAssignbatch_normalization/gamma_5,batch_normalization/gamma_5/Initializer/ones*.
_class$
" loc:@batch_normalization/gamma_5*
validate_shape(*
use_locking(*
T0
�
 batch_normalization/gamma_5/readIdentitybatch_normalization/gamma_5*
T0*.
_class$
" loc:@batch_normalization/gamma_5
�
<batch_normalization/beta_5/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@batch_normalization/beta_5*
valueB:@*
dtype0
�
2batch_normalization/beta_5/Initializer/zeros/ConstConst*-
_class#
!loc:@batch_normalization/beta_5*
valueB
 *    *
dtype0
�
,batch_normalization/beta_5/Initializer/zerosFill<batch_normalization/beta_5/Initializer/zeros/shape_as_tensor2batch_normalization/beta_5/Initializer/zeros/Const*
T0*-
_class#
!loc:@batch_normalization/beta_5*

index_type0
�
batch_normalization/beta_5
VariableV2*
shape:@*
shared_name *-
_class#
!loc:@batch_normalization/beta_5*
dtype0*
	container 
�
!batch_normalization/beta_5/AssignAssignbatch_normalization/beta_5,batch_normalization/beta_5/Initializer/zeros*
T0*-
_class#
!loc:@batch_normalization/beta_5*
validate_shape(*
use_locking(

batch_normalization/beta_5/readIdentitybatch_normalization/beta_5*
T0*-
_class#
!loc:@batch_normalization/beta_5
�
Cbatch_normalization/moving_mean_5/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@batch_normalization/moving_mean_5*
valueB:@*
dtype0
�
9batch_normalization/moving_mean_5/Initializer/zeros/ConstConst*4
_class*
(&loc:@batch_normalization/moving_mean_5*
valueB
 *    *
dtype0
�
3batch_normalization/moving_mean_5/Initializer/zerosFillCbatch_normalization/moving_mean_5/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_5/Initializer/zeros/Const*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_5*

index_type0
�
!batch_normalization/moving_mean_5
VariableV2*
shape:@*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_5*
dtype0*
	container 
�
(batch_normalization/moving_mean_5/AssignAssign!batch_normalization/moving_mean_53batch_normalization/moving_mean_5/Initializer/zeros*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_5*
validate_shape(*
use_locking(
�
&batch_normalization/moving_mean_5/readIdentity!batch_normalization/moving_mean_5*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_5
�
Fbatch_normalization/moving_variance_5/Initializer/ones/shape_as_tensorConst*8
_class.
,*loc:@batch_normalization/moving_variance_5*
valueB:@*
dtype0
�
<batch_normalization/moving_variance_5/Initializer/ones/ConstConst*8
_class.
,*loc:@batch_normalization/moving_variance_5*
valueB
 *  �?*
dtype0
�
6batch_normalization/moving_variance_5/Initializer/onesFillFbatch_normalization/moving_variance_5/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_5/Initializer/ones/Const*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_5*

index_type0
�
%batch_normalization/moving_variance_5
VariableV2*
	container *
shape:@*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_5*
dtype0
�
,batch_normalization/moving_variance_5/AssignAssign%batch_normalization/moving_variance_56batch_normalization/moving_variance_5/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_5*
validate_shape(
�
*batch_normalization/moving_variance_5/readIdentity%batch_normalization/moving_variance_5*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_5
�
$batch_normalization/FusedBatchNorm_5FusedBatchNormres2b_branch2a batch_normalization/gamma_5/readbatch_normalization/beta_5/read&batch_normalization/moving_mean_5/read*batch_normalization/moving_variance_5/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
H
batch_normalization/Const_5Const*
valueB
 * �:*
dtype0
F
bn2b_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2b_branch2aIdentitybn2b_branch2a*
T0
6
res2b_branch2a_reluReluscale2b_branch2a*
T0
J
res2b_branch2b/kernelConst*
valueB@@*
dtype0
�
res2b_branch2bConv2Dres2b_branch2a_relures2b_branch2b/kernel*
paddingSAME*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
<batch_normalization/gamma_6/Initializer/ones/shape_as_tensorConst*
valueB:@*.
_class$
" loc:@batch_normalization/gamma_6*
dtype0
�
2batch_normalization/gamma_6/Initializer/ones/ConstConst*
valueB
 *  �?*.
_class$
" loc:@batch_normalization/gamma_6*
dtype0
�
,batch_normalization/gamma_6/Initializer/onesFill<batch_normalization/gamma_6/Initializer/ones/shape_as_tensor2batch_normalization/gamma_6/Initializer/ones/Const*

index_type0*.
_class$
" loc:@batch_normalization/gamma_6*
T0
�
batch_normalization/gamma_6
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/gamma_6*
dtype0*
	container *
shape:@
�
"batch_normalization/gamma_6/AssignAssignbatch_normalization/gamma_6,batch_normalization/gamma_6/Initializer/ones*
T0*.
_class$
" loc:@batch_normalization/gamma_6*
validate_shape(*
use_locking(
�
 batch_normalization/gamma_6/readIdentitybatch_normalization/gamma_6*
T0*.
_class$
" loc:@batch_normalization/gamma_6
�
<batch_normalization/beta_6/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization/beta_6*
dtype0
�
2batch_normalization/beta_6/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization/beta_6*
dtype0
�
,batch_normalization/beta_6/Initializer/zerosFill<batch_normalization/beta_6/Initializer/zeros/shape_as_tensor2batch_normalization/beta_6/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization/beta_6
�
batch_normalization/beta_6
VariableV2*-
_class#
!loc:@batch_normalization/beta_6*
dtype0*
	container *
shape:@*
shared_name 
�
!batch_normalization/beta_6/AssignAssignbatch_normalization/beta_6,batch_normalization/beta_6/Initializer/zeros*
T0*-
_class#
!loc:@batch_normalization/beta_6*
validate_shape(*
use_locking(

batch_normalization/beta_6/readIdentitybatch_normalization/beta_6*
T0*-
_class#
!loc:@batch_normalization/beta_6
�
Cbatch_normalization/moving_mean_6/Initializer/zeros/shape_as_tensorConst*
valueB:@*4
_class*
(&loc:@batch_normalization/moving_mean_6*
dtype0
�
9batch_normalization/moving_mean_6/Initializer/zeros/ConstConst*
valueB
 *    *4
_class*
(&loc:@batch_normalization/moving_mean_6*
dtype0
�
3batch_normalization/moving_mean_6/Initializer/zerosFillCbatch_normalization/moving_mean_6/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_6/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization/moving_mean_6
�
!batch_normalization/moving_mean_6
VariableV2*
	container *
shape:@*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_6*
dtype0
�
(batch_normalization/moving_mean_6/AssignAssign!batch_normalization/moving_mean_63batch_normalization/moving_mean_6/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_6*
validate_shape(
�
&batch_normalization/moving_mean_6/readIdentity!batch_normalization/moving_mean_6*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_6
�
Fbatch_normalization/moving_variance_6/Initializer/ones/shape_as_tensorConst*
valueB:@*8
_class.
,*loc:@batch_normalization/moving_variance_6*
dtype0
�
<batch_normalization/moving_variance_6/Initializer/ones/ConstConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization/moving_variance_6*
dtype0
�
6batch_normalization/moving_variance_6/Initializer/onesFillFbatch_normalization/moving_variance_6/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_6/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization/moving_variance_6
�
%batch_normalization/moving_variance_6
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_6*
dtype0*
	container *
shape:@
�
,batch_normalization/moving_variance_6/AssignAssign%batch_normalization/moving_variance_66batch_normalization/moving_variance_6/Initializer/ones*8
_class.
,*loc:@batch_normalization/moving_variance_6*
validate_shape(*
use_locking(*
T0
�
*batch_normalization/moving_variance_6/readIdentity%batch_normalization/moving_variance_6*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_6
�
$batch_normalization/FusedBatchNorm_6FusedBatchNormres2b_branch2b batch_normalization/gamma_6/readbatch_normalization/beta_6/read&batch_normalization/moving_mean_6/read*batch_normalization/moving_variance_6/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
H
batch_normalization/Const_6Const*
dtype0*
valueB
 * �:
F
bn2b_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2b_branch2bIdentitybn2b_branch2b*
T0
6
res2b_branch2b_reluReluscale2b_branch2b*
T0
K
res2b_branch2c/kernelConst*
valueB@�*
dtype0
�
res2b_branch2cConv2Dres2b_branch2b_relures2b_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
<batch_normalization/gamma_7/Initializer/ones/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/gamma_7*
valueB:�*
dtype0
�
2batch_normalization/gamma_7/Initializer/ones/ConstConst*.
_class$
" loc:@batch_normalization/gamma_7*
valueB
 *  �?*
dtype0
�
,batch_normalization/gamma_7/Initializer/onesFill<batch_normalization/gamma_7/Initializer/ones/shape_as_tensor2batch_normalization/gamma_7/Initializer/ones/Const*
T0*.
_class$
" loc:@batch_normalization/gamma_7*

index_type0
�
batch_normalization/gamma_7
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/gamma_7*
dtype0*
	container 
�
"batch_normalization/gamma_7/AssignAssignbatch_normalization/gamma_7,batch_normalization/gamma_7/Initializer/ones*
T0*.
_class$
" loc:@batch_normalization/gamma_7*
validate_shape(*
use_locking(
�
 batch_normalization/gamma_7/readIdentitybatch_normalization/gamma_7*
T0*.
_class$
" loc:@batch_normalization/gamma_7
�
<batch_normalization/beta_7/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@batch_normalization/beta_7*
valueB:�*
dtype0
�
2batch_normalization/beta_7/Initializer/zeros/ConstConst*-
_class#
!loc:@batch_normalization/beta_7*
valueB
 *    *
dtype0
�
,batch_normalization/beta_7/Initializer/zerosFill<batch_normalization/beta_7/Initializer/zeros/shape_as_tensor2batch_normalization/beta_7/Initializer/zeros/Const*
T0*-
_class#
!loc:@batch_normalization/beta_7*

index_type0
�
batch_normalization/beta_7
VariableV2*
shape:�*
shared_name *-
_class#
!loc:@batch_normalization/beta_7*
dtype0*
	container 
�
!batch_normalization/beta_7/AssignAssignbatch_normalization/beta_7,batch_normalization/beta_7/Initializer/zeros*
validate_shape(*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_7

batch_normalization/beta_7/readIdentitybatch_normalization/beta_7*
T0*-
_class#
!loc:@batch_normalization/beta_7
�
Cbatch_normalization/moving_mean_7/Initializer/zeros/shape_as_tensorConst*
dtype0*4
_class*
(&loc:@batch_normalization/moving_mean_7*
valueB:�
�
9batch_normalization/moving_mean_7/Initializer/zeros/ConstConst*4
_class*
(&loc:@batch_normalization/moving_mean_7*
valueB
 *    *
dtype0
�
3batch_normalization/moving_mean_7/Initializer/zerosFillCbatch_normalization/moving_mean_7/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_7/Initializer/zeros/Const*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_7*

index_type0
�
!batch_normalization/moving_mean_7
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_7*
dtype0*
	container *
shape:�
�
(batch_normalization/moving_mean_7/AssignAssign!batch_normalization/moving_mean_73batch_normalization/moving_mean_7/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_7*
validate_shape(
�
&batch_normalization/moving_mean_7/readIdentity!batch_normalization/moving_mean_7*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_7
�
Fbatch_normalization/moving_variance_7/Initializer/ones/shape_as_tensorConst*8
_class.
,*loc:@batch_normalization/moving_variance_7*
valueB:�*
dtype0
�
<batch_normalization/moving_variance_7/Initializer/ones/ConstConst*8
_class.
,*loc:@batch_normalization/moving_variance_7*
valueB
 *  �?*
dtype0
�
6batch_normalization/moving_variance_7/Initializer/onesFillFbatch_normalization/moving_variance_7/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_7/Initializer/ones/Const*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_7*

index_type0
�
%batch_normalization/moving_variance_7
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_7*
dtype0*
	container *
shape:�
�
,batch_normalization/moving_variance_7/AssignAssign%batch_normalization/moving_variance_76batch_normalization/moving_variance_7/Initializer/ones*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_7*
validate_shape(*
use_locking(
�
*batch_normalization/moving_variance_7/readIdentity%batch_normalization/moving_variance_7*8
_class.
,*loc:@batch_normalization/moving_variance_7*
T0
�
$batch_normalization/FusedBatchNorm_7FusedBatchNormres2b_branch2c batch_normalization/gamma_7/readbatch_normalization/beta_7/read&batch_normalization/moving_mean_7/read*batch_normalization/moving_variance_7/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
H
batch_normalization/Const_7Const*
valueB
 * �:*
dtype0
F
bn2b_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2b_branch2cIdentitybn2b_branch2c*
T0
3
res2bAdd
res2a_reluscale2b_branch2c*
T0
"

res2b_reluRelures2b*
T0
J
res2c_branch2a/kernelConst*
dtype0*
valueB@@
�
res2c_branch2aConv2D
res2b_relures2c_branch2a/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
<batch_normalization/gamma_8/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:@*.
_class$
" loc:@batch_normalization/gamma_8
�
2batch_normalization/gamma_8/Initializer/ones/ConstConst*
valueB
 *  �?*.
_class$
" loc:@batch_normalization/gamma_8*
dtype0
�
,batch_normalization/gamma_8/Initializer/onesFill<batch_normalization/gamma_8/Initializer/ones/shape_as_tensor2batch_normalization/gamma_8/Initializer/ones/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/gamma_8
�
batch_normalization/gamma_8
VariableV2*
shape:@*
shared_name *.
_class$
" loc:@batch_normalization/gamma_8*
dtype0*
	container 
�
"batch_normalization/gamma_8/AssignAssignbatch_normalization/gamma_8,batch_normalization/gamma_8/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_8*
validate_shape(
�
 batch_normalization/gamma_8/readIdentitybatch_normalization/gamma_8*
T0*.
_class$
" loc:@batch_normalization/gamma_8
�
<batch_normalization/beta_8/Initializer/zeros/shape_as_tensorConst*
valueB:@*-
_class#
!loc:@batch_normalization/beta_8*
dtype0
�
2batch_normalization/beta_8/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@batch_normalization/beta_8*
dtype0
�
,batch_normalization/beta_8/Initializer/zerosFill<batch_normalization/beta_8/Initializer/zeros/shape_as_tensor2batch_normalization/beta_8/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@batch_normalization/beta_8
�
batch_normalization/beta_8
VariableV2*
shape:@*
shared_name *-
_class#
!loc:@batch_normalization/beta_8*
dtype0*
	container 
�
!batch_normalization/beta_8/AssignAssignbatch_normalization/beta_8,batch_normalization/beta_8/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_8*
validate_shape(

batch_normalization/beta_8/readIdentitybatch_normalization/beta_8*-
_class#
!loc:@batch_normalization/beta_8*
T0
�
Cbatch_normalization/moving_mean_8/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:@*4
_class*
(&loc:@batch_normalization/moving_mean_8
�
9batch_normalization/moving_mean_8/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *4
_class*
(&loc:@batch_normalization/moving_mean_8
�
3batch_normalization/moving_mean_8/Initializer/zerosFillCbatch_normalization/moving_mean_8/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_8/Initializer/zeros/Const*
T0*

index_type0*4
_class*
(&loc:@batch_normalization/moving_mean_8
�
!batch_normalization/moving_mean_8
VariableV2*
shared_name *4
_class*
(&loc:@batch_normalization/moving_mean_8*
dtype0*
	container *
shape:@
�
(batch_normalization/moving_mean_8/AssignAssign!batch_normalization/moving_mean_83batch_normalization/moving_mean_8/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_8*
validate_shape(
�
&batch_normalization/moving_mean_8/readIdentity!batch_normalization/moving_mean_8*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_8
�
Fbatch_normalization/moving_variance_8/Initializer/ones/shape_as_tensorConst*
valueB:@*8
_class.
,*loc:@batch_normalization/moving_variance_8*
dtype0
�
<batch_normalization/moving_variance_8/Initializer/ones/ConstConst*
valueB
 *  �?*8
_class.
,*loc:@batch_normalization/moving_variance_8*
dtype0
�
6batch_normalization/moving_variance_8/Initializer/onesFillFbatch_normalization/moving_variance_8/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_8/Initializer/ones/Const*
T0*

index_type0*8
_class.
,*loc:@batch_normalization/moving_variance_8
�
%batch_normalization/moving_variance_8
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_8*
dtype0*
	container *
shape:@
�
,batch_normalization/moving_variance_8/AssignAssign%batch_normalization/moving_variance_86batch_normalization/moving_variance_8/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_8*
validate_shape(
�
*batch_normalization/moving_variance_8/readIdentity%batch_normalization/moving_variance_8*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_8
�
$batch_normalization/FusedBatchNorm_8FusedBatchNormres2c_branch2a batch_normalization/gamma_8/readbatch_normalization/beta_8/read&batch_normalization/moving_mean_8/read*batch_normalization/moving_variance_8/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
H
batch_normalization/Const_8Const*
valueB
 * �:*
dtype0
F
bn2c_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2c_branch2aIdentitybn2c_branch2a*
T0
6
res2c_branch2a_reluReluscale2c_branch2a*
T0
J
res2c_branch2b/kernelConst*
valueB@@*
dtype0
�
res2c_branch2bConv2Dres2c_branch2a_relures2c_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
<batch_normalization/gamma_9/Initializer/ones/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/gamma_9*
valueB:@*
dtype0
�
2batch_normalization/gamma_9/Initializer/ones/ConstConst*.
_class$
" loc:@batch_normalization/gamma_9*
valueB
 *  �?*
dtype0
�
,batch_normalization/gamma_9/Initializer/onesFill<batch_normalization/gamma_9/Initializer/ones/shape_as_tensor2batch_normalization/gamma_9/Initializer/ones/Const*.
_class$
" loc:@batch_normalization/gamma_9*

index_type0*
T0
�
batch_normalization/gamma_9
VariableV2*.
_class$
" loc:@batch_normalization/gamma_9*
dtype0*
	container *
shape:@*
shared_name 
�
"batch_normalization/gamma_9/AssignAssignbatch_normalization/gamma_9,batch_normalization/gamma_9/Initializer/ones*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/gamma_9*
validate_shape(
�
 batch_normalization/gamma_9/readIdentitybatch_normalization/gamma_9*.
_class$
" loc:@batch_normalization/gamma_9*
T0
�
<batch_normalization/beta_9/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@batch_normalization/beta_9*
valueB:@*
dtype0
�
2batch_normalization/beta_9/Initializer/zeros/ConstConst*-
_class#
!loc:@batch_normalization/beta_9*
valueB
 *    *
dtype0
�
,batch_normalization/beta_9/Initializer/zerosFill<batch_normalization/beta_9/Initializer/zeros/shape_as_tensor2batch_normalization/beta_9/Initializer/zeros/Const*
T0*-
_class#
!loc:@batch_normalization/beta_9*

index_type0
�
batch_normalization/beta_9
VariableV2*-
_class#
!loc:@batch_normalization/beta_9*
dtype0*
	container *
shape:@*
shared_name 
�
!batch_normalization/beta_9/AssignAssignbatch_normalization/beta_9,batch_normalization/beta_9/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@batch_normalization/beta_9*
validate_shape(

batch_normalization/beta_9/readIdentitybatch_normalization/beta_9*
T0*-
_class#
!loc:@batch_normalization/beta_9
�
Cbatch_normalization/moving_mean_9/Initializer/zeros/shape_as_tensorConst*4
_class*
(&loc:@batch_normalization/moving_mean_9*
valueB:@*
dtype0
�
9batch_normalization/moving_mean_9/Initializer/zeros/ConstConst*4
_class*
(&loc:@batch_normalization/moving_mean_9*
valueB
 *    *
dtype0
�
3batch_normalization/moving_mean_9/Initializer/zerosFillCbatch_normalization/moving_mean_9/Initializer/zeros/shape_as_tensor9batch_normalization/moving_mean_9/Initializer/zeros/Const*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_9*

index_type0
�
!batch_normalization/moving_mean_9
VariableV2*4
_class*
(&loc:@batch_normalization/moving_mean_9*
dtype0*
	container *
shape:@*
shared_name 
�
(batch_normalization/moving_mean_9/AssignAssign!batch_normalization/moving_mean_93batch_normalization/moving_mean_9/Initializer/zeros*4
_class*
(&loc:@batch_normalization/moving_mean_9*
validate_shape(*
use_locking(*
T0
�
&batch_normalization/moving_mean_9/readIdentity!batch_normalization/moving_mean_9*
T0*4
_class*
(&loc:@batch_normalization/moving_mean_9
�
Fbatch_normalization/moving_variance_9/Initializer/ones/shape_as_tensorConst*8
_class.
,*loc:@batch_normalization/moving_variance_9*
valueB:@*
dtype0
�
<batch_normalization/moving_variance_9/Initializer/ones/ConstConst*8
_class.
,*loc:@batch_normalization/moving_variance_9*
valueB
 *  �?*
dtype0
�
6batch_normalization/moving_variance_9/Initializer/onesFillFbatch_normalization/moving_variance_9/Initializer/ones/shape_as_tensor<batch_normalization/moving_variance_9/Initializer/ones/Const*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_9*

index_type0
�
%batch_normalization/moving_variance_9
VariableV2*
shared_name *8
_class.
,*loc:@batch_normalization/moving_variance_9*
dtype0*
	container *
shape:@
�
,batch_normalization/moving_variance_9/AssignAssign%batch_normalization/moving_variance_96batch_normalization/moving_variance_9/Initializer/ones*
use_locking(*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_9*
validate_shape(
�
*batch_normalization/moving_variance_9/readIdentity%batch_normalization/moving_variance_9*
T0*8
_class.
,*loc:@batch_normalization/moving_variance_9
�
$batch_normalization/FusedBatchNorm_9FusedBatchNormres2c_branch2b batch_normalization/gamma_9/readbatch_normalization/beta_9/read&batch_normalization/moving_mean_9/read*batch_normalization/moving_variance_9/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
H
batch_normalization/Const_9Const*
valueB
 * �:*
dtype0
F
bn2c_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2c_branch2bIdentitybn2c_branch2b*
T0
6
res2c_branch2b_reluReluscale2c_branch2b*
T0
K
res2c_branch2c/kernelConst*
valueB@�*
dtype0
�
res2c_branch2cConv2Dres2c_branch2b_relures2c_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_10/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_10*
dtype0
�
3batch_normalization/gamma_10/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_10*
dtype0
�
-batch_normalization/gamma_10/Initializer/onesFill=batch_normalization/gamma_10/Initializer/ones/shape_as_tensor3batch_normalization/gamma_10/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_10
�
batch_normalization/gamma_10
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_10*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_10/AssignAssignbatch_normalization/gamma_10-batch_normalization/gamma_10/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_10*
validate_shape(
�
!batch_normalization/gamma_10/readIdentitybatch_normalization/gamma_10*
T0*/
_class%
#!loc:@batch_normalization/gamma_10
�
=batch_normalization/beta_10/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_10*
dtype0
�
3batch_normalization/beta_10/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_10*
dtype0
�
-batch_normalization/beta_10/Initializer/zerosFill=batch_normalization/beta_10/Initializer/zeros/shape_as_tensor3batch_normalization/beta_10/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_10
�
batch_normalization/beta_10
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_10*
dtype0*
	container 
�
"batch_normalization/beta_10/AssignAssignbatch_normalization/beta_10-batch_normalization/beta_10/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_10*
validate_shape(
�
 batch_normalization/beta_10/readIdentitybatch_normalization/beta_10*
T0*.
_class$
" loc:@batch_normalization/beta_10
�
Dbatch_normalization/moving_mean_10/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_10*
dtype0
�
:batch_normalization/moving_mean_10/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_10
�
4batch_normalization/moving_mean_10/Initializer/zerosFillDbatch_normalization/moving_mean_10/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_10/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_10
�
"batch_normalization/moving_mean_10
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_10*
dtype0*
	container 
�
)batch_normalization/moving_mean_10/AssignAssign"batch_normalization/moving_mean_104batch_normalization/moving_mean_10/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_10*
validate_shape(
�
'batch_normalization/moving_mean_10/readIdentity"batch_normalization/moving_mean_10*5
_class+
)'loc:@batch_normalization/moving_mean_10*
T0
�
Gbatch_normalization/moving_variance_10/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_10*
dtype0
�
=batch_normalization/moving_variance_10/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_10*
dtype0
�
7batch_normalization/moving_variance_10/Initializer/onesFillGbatch_normalization/moving_variance_10/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_10/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_10
�
&batch_normalization/moving_variance_10
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_10*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_10/AssignAssign&batch_normalization/moving_variance_107batch_normalization/moving_variance_10/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_10*
validate_shape(
�
+batch_normalization/moving_variance_10/readIdentity&batch_normalization/moving_variance_10*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_10
�
%batch_normalization/FusedBatchNorm_10FusedBatchNormres2c_branch2c!batch_normalization/gamma_10/read batch_normalization/beta_10/read'batch_normalization/moving_mean_10/read+batch_normalization/moving_variance_10/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_10Const*
valueB
 * �:*
dtype0
F
bn2c_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale2c_branch2cIdentitybn2c_branch2c*
T0
3
res2cAdd
res2b_reluscale2c_branch2c*
T0
"

res2c_reluRelures2c*
T0
J
res3a_branch1/kernelConst*
valueB@�*
dtype0
�
res3a_branch1Conv2D
res2c_relures3a_branch1/kernel*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_11/Initializer/ones/shape_as_tensorConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_11*
valueB:�
�
3batch_normalization/gamma_11/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_11*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_11/Initializer/onesFill=batch_normalization/gamma_11/Initializer/ones/shape_as_tensor3batch_normalization/gamma_11/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_11*

index_type0
�
batch_normalization/gamma_11
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_11*
dtype0*
	container 
�
#batch_normalization/gamma_11/AssignAssignbatch_normalization/gamma_11-batch_normalization/gamma_11/Initializer/ones*/
_class%
#!loc:@batch_normalization/gamma_11*
validate_shape(*
use_locking(*
T0
�
!batch_normalization/gamma_11/readIdentitybatch_normalization/gamma_11*
T0*/
_class%
#!loc:@batch_normalization/gamma_11
�
=batch_normalization/beta_11/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_11*
valueB:�*
dtype0
�
3batch_normalization/beta_11/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_11*
valueB
 *    *
dtype0
�
-batch_normalization/beta_11/Initializer/zerosFill=batch_normalization/beta_11/Initializer/zeros/shape_as_tensor3batch_normalization/beta_11/Initializer/zeros/Const*.
_class$
" loc:@batch_normalization/beta_11*

index_type0*
T0
�
batch_normalization/beta_11
VariableV2*.
_class$
" loc:@batch_normalization/beta_11*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_11/AssignAssignbatch_normalization/beta_11-batch_normalization/beta_11/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_11*
validate_shape(*
use_locking(
�
 batch_normalization/beta_11/readIdentitybatch_normalization/beta_11*.
_class$
" loc:@batch_normalization/beta_11*
T0
�
Dbatch_normalization/moving_mean_11/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_11*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_11/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_11*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_11/Initializer/zerosFillDbatch_normalization/moving_mean_11/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_11/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_11*

index_type0
�
"batch_normalization/moving_mean_11
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_11*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_11/AssignAssign"batch_normalization/moving_mean_114batch_normalization/moving_mean_11/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_11*
validate_shape(
�
'batch_normalization/moving_mean_11/readIdentity"batch_normalization/moving_mean_11*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_11
�
Gbatch_normalization/moving_variance_11/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_11*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_11/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_11*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_11/Initializer/onesFillGbatch_normalization/moving_variance_11/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_11/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_11*

index_type0
�
&batch_normalization/moving_variance_11
VariableV2*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_11*
dtype0
�
-batch_normalization/moving_variance_11/AssignAssign&batch_normalization/moving_variance_117batch_normalization/moving_variance_11/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_11*
validate_shape(
�
+batch_normalization/moving_variance_11/readIdentity&batch_normalization/moving_variance_11*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_11
�
%batch_normalization/FusedBatchNorm_11FusedBatchNormres3a_branch1!batch_normalization/gamma_11/read batch_normalization/beta_11/read'batch_normalization/moving_mean_11/read+batch_normalization/moving_variance_11/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_11Const*
valueB
 * �:*
dtype0
E
bn3a_branch1Identity"batch_normalization/FusedBatchNorm*
T0
2
scale3a_branch1Identitybn3a_branch1*
T0
K
res3a_branch2a/kernelConst*
dtype0*
valueB@�
�
res3a_branch2aConv2D
res2c_relures3a_branch2a/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_12/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_12*
dtype0
�
3batch_normalization/gamma_12/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_12*
dtype0
�
-batch_normalization/gamma_12/Initializer/onesFill=batch_normalization/gamma_12/Initializer/ones/shape_as_tensor3batch_normalization/gamma_12/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_12
�
batch_normalization/gamma_12
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_12*
dtype0
�
#batch_normalization/gamma_12/AssignAssignbatch_normalization/gamma_12-batch_normalization/gamma_12/Initializer/ones*
T0*/
_class%
#!loc:@batch_normalization/gamma_12*
validate_shape(*
use_locking(
�
!batch_normalization/gamma_12/readIdentitybatch_normalization/gamma_12*
T0*/
_class%
#!loc:@batch_normalization/gamma_12
�
=batch_normalization/beta_12/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_12*
dtype0
�
3batch_normalization/beta_12/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_12*
dtype0
�
-batch_normalization/beta_12/Initializer/zerosFill=batch_normalization/beta_12/Initializer/zeros/shape_as_tensor3batch_normalization/beta_12/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_12
�
batch_normalization/beta_12
VariableV2*.
_class$
" loc:@batch_normalization/beta_12*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_12/AssignAssignbatch_normalization/beta_12-batch_normalization/beta_12/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_12
�
 batch_normalization/beta_12/readIdentitybatch_normalization/beta_12*
T0*.
_class$
" loc:@batch_normalization/beta_12
�
Dbatch_normalization/moving_mean_12/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_12*
dtype0
�
:batch_normalization/moving_mean_12/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_12*
dtype0
�
4batch_normalization/moving_mean_12/Initializer/zerosFillDbatch_normalization/moving_mean_12/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_12/Initializer/zeros/Const*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_12*
T0
�
"batch_normalization/moving_mean_12
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_12*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_12/AssignAssign"batch_normalization/moving_mean_124batch_normalization/moving_mean_12/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_12*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_12/readIdentity"batch_normalization/moving_mean_12*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_12
�
Gbatch_normalization/moving_variance_12/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_12*
dtype0
�
=batch_normalization/moving_variance_12/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_12*
dtype0
�
7batch_normalization/moving_variance_12/Initializer/onesFillGbatch_normalization/moving_variance_12/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_12/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_12
�
&batch_normalization/moving_variance_12
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_12*
dtype0*
	container 
�
-batch_normalization/moving_variance_12/AssignAssign&batch_normalization/moving_variance_127batch_normalization/moving_variance_12/Initializer/ones*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_12
�
+batch_normalization/moving_variance_12/readIdentity&batch_normalization/moving_variance_12*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_12
�
%batch_normalization/FusedBatchNorm_12FusedBatchNormres3a_branch2a!batch_normalization/gamma_12/read batch_normalization/beta_12/read'batch_normalization/moving_mean_12/read+batch_normalization/moving_variance_12/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_12Const*
valueB
 * �:*
dtype0
F
bn3a_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3a_branch2aIdentitybn3a_branch2a*
T0
6
res3a_branch2a_reluReluscale3a_branch2a*
T0
K
res3a_branch2b/kernelConst*
valueB@�*
dtype0
�
res3a_branch2bConv2Dres3a_branch2a_relures3a_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_13/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_13*
valueB:�*
dtype0
�
3batch_normalization/gamma_13/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_13*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_13/Initializer/onesFill=batch_normalization/gamma_13/Initializer/ones/shape_as_tensor3batch_normalization/gamma_13/Initializer/ones/Const*/
_class%
#!loc:@batch_normalization/gamma_13*

index_type0*
T0
�
batch_normalization/gamma_13
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_13*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_13/AssignAssignbatch_normalization/gamma_13-batch_normalization/gamma_13/Initializer/ones*
validate_shape(*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_13
�
!batch_normalization/gamma_13/readIdentitybatch_normalization/gamma_13*
T0*/
_class%
#!loc:@batch_normalization/gamma_13
�
=batch_normalization/beta_13/Initializer/zeros/shape_as_tensorConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_13*
valueB:�
�
3batch_normalization/beta_13/Initializer/zeros/ConstConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_13*
valueB
 *    
�
-batch_normalization/beta_13/Initializer/zerosFill=batch_normalization/beta_13/Initializer/zeros/shape_as_tensor3batch_normalization/beta_13/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_13*

index_type0
�
batch_normalization/beta_13
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_13
�
"batch_normalization/beta_13/AssignAssignbatch_normalization/beta_13-batch_normalization/beta_13/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_13*
validate_shape(
�
 batch_normalization/beta_13/readIdentitybatch_normalization/beta_13*
T0*.
_class$
" loc:@batch_normalization/beta_13
�
Dbatch_normalization/moving_mean_13/Initializer/zeros/shape_as_tensorConst*
dtype0*5
_class+
)'loc:@batch_normalization/moving_mean_13*
valueB:�
�
:batch_normalization/moving_mean_13/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_13*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_13/Initializer/zerosFillDbatch_normalization/moving_mean_13/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_13/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_13*

index_type0
�
"batch_normalization/moving_mean_13
VariableV2*
dtype0*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_13
�
)batch_normalization/moving_mean_13/AssignAssign"batch_normalization/moving_mean_134batch_normalization/moving_mean_13/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_13*
validate_shape(
�
'batch_normalization/moving_mean_13/readIdentity"batch_normalization/moving_mean_13*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_13
�
Gbatch_normalization/moving_variance_13/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_13*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_13/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_13*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_13/Initializer/onesFillGbatch_normalization/moving_variance_13/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_13/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_13*

index_type0
�
&batch_normalization/moving_variance_13
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_13*
dtype0*
	container 
�
-batch_normalization/moving_variance_13/AssignAssign&batch_normalization/moving_variance_137batch_normalization/moving_variance_13/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_13*
validate_shape(
�
+batch_normalization/moving_variance_13/readIdentity&batch_normalization/moving_variance_13*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_13
�
%batch_normalization/FusedBatchNorm_13FusedBatchNormres3a_branch2b!batch_normalization/gamma_13/read batch_normalization/beta_13/read'batch_normalization/moving_mean_13/read+batch_normalization/moving_variance_13/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_13Const*
valueB
 * �:*
dtype0
F
bn3a_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3a_branch2bIdentitybn3a_branch2b*
T0
6
res3a_branch2b_reluReluscale3a_branch2b*
T0
K
res3a_branch2c/kernelConst*
valueB@�*
dtype0
�
res3a_branch2cConv2Dres3a_branch2b_relures3a_branch2c/kernel*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

�
=batch_normalization/gamma_14/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_14*
dtype0
�
3batch_normalization/gamma_14/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_14*
dtype0
�
-batch_normalization/gamma_14/Initializer/onesFill=batch_normalization/gamma_14/Initializer/ones/shape_as_tensor3batch_normalization/gamma_14/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_14
�
batch_normalization/gamma_14
VariableV2*
dtype0*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_14
�
#batch_normalization/gamma_14/AssignAssignbatch_normalization/gamma_14-batch_normalization/gamma_14/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_14*
validate_shape(
�
!batch_normalization/gamma_14/readIdentitybatch_normalization/gamma_14*
T0*/
_class%
#!loc:@batch_normalization/gamma_14
�
=batch_normalization/beta_14/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_14*
dtype0
�
3batch_normalization/beta_14/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_14
�
-batch_normalization/beta_14/Initializer/zerosFill=batch_normalization/beta_14/Initializer/zeros/shape_as_tensor3batch_normalization/beta_14/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_14
�
batch_normalization/beta_14
VariableV2*.
_class$
" loc:@batch_normalization/beta_14*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_14/AssignAssignbatch_normalization/beta_14-batch_normalization/beta_14/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_14*
validate_shape(
�
 batch_normalization/beta_14/readIdentitybatch_normalization/beta_14*
T0*.
_class$
" loc:@batch_normalization/beta_14
�
Dbatch_normalization/moving_mean_14/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_14*
dtype0
�
:batch_normalization/moving_mean_14/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_14*
dtype0
�
4batch_normalization/moving_mean_14/Initializer/zerosFillDbatch_normalization/moving_mean_14/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_14/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_14
�
"batch_normalization/moving_mean_14
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_14*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_14/AssignAssign"batch_normalization/moving_mean_144batch_normalization/moving_mean_14/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_14*
validate_shape(
�
'batch_normalization/moving_mean_14/readIdentity"batch_normalization/moving_mean_14*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_14
�
Gbatch_normalization/moving_variance_14/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_14*
dtype0
�
=batch_normalization/moving_variance_14/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_14*
dtype0
�
7batch_normalization/moving_variance_14/Initializer/onesFillGbatch_normalization/moving_variance_14/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_14/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_14
�
&batch_normalization/moving_variance_14
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_14*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_14/AssignAssign&batch_normalization/moving_variance_147batch_normalization/moving_variance_14/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_14*
validate_shape(
�
+batch_normalization/moving_variance_14/readIdentity&batch_normalization/moving_variance_14*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_14
�
%batch_normalization/FusedBatchNorm_14FusedBatchNormres3a_branch2c!batch_normalization/gamma_14/read batch_normalization/beta_14/read'batch_normalization/moving_mean_14/read+batch_normalization/moving_variance_14/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
I
batch_normalization/Const_14Const*
valueB
 * �:*
dtype0
F
bn3a_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3a_branch2cIdentitybn3a_branch2c*
T0
8
res3aAddscale3a_branch1scale3a_branch2c*
T0
"

res3a_reluRelures3a*
T0
K
res3b_branch2a/kernelConst*
valueB@�*
dtype0
�
res3b_branch2aConv2D
res3a_relures3b_branch2a/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_15/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_15*
valueB:�*
dtype0
�
3batch_normalization/gamma_15/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_15*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_15/Initializer/onesFill=batch_normalization/gamma_15/Initializer/ones/shape_as_tensor3batch_normalization/gamma_15/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_15*

index_type0
�
batch_normalization/gamma_15
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_15*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_15/AssignAssignbatch_normalization/gamma_15-batch_normalization/gamma_15/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_15*
validate_shape(
�
!batch_normalization/gamma_15/readIdentitybatch_normalization/gamma_15*
T0*/
_class%
#!loc:@batch_normalization/gamma_15
�
=batch_normalization/beta_15/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_15*
valueB:�*
dtype0
�
3batch_normalization/beta_15/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_15*
valueB
 *    *
dtype0
�
-batch_normalization/beta_15/Initializer/zerosFill=batch_normalization/beta_15/Initializer/zeros/shape_as_tensor3batch_normalization/beta_15/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_15*

index_type0
�
batch_normalization/beta_15
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_15*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_15/AssignAssignbatch_normalization/beta_15-batch_normalization/beta_15/Initializer/zeros*.
_class$
" loc:@batch_normalization/beta_15*
validate_shape(*
use_locking(*
T0
�
 batch_normalization/beta_15/readIdentitybatch_normalization/beta_15*
T0*.
_class$
" loc:@batch_normalization/beta_15
�
Dbatch_normalization/moving_mean_15/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_15*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_15/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_15*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_15/Initializer/zerosFillDbatch_normalization/moving_mean_15/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_15/Initializer/zeros/Const*5
_class+
)'loc:@batch_normalization/moving_mean_15*

index_type0*
T0
�
"batch_normalization/moving_mean_15
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_15*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_15/AssignAssign"batch_normalization/moving_mean_154batch_normalization/moving_mean_15/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_15
�
'batch_normalization/moving_mean_15/readIdentity"batch_normalization/moving_mean_15*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_15
�
Gbatch_normalization/moving_variance_15/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_15*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_15/Initializer/ones/ConstConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_15*
valueB
 *  �?
�
7batch_normalization/moving_variance_15/Initializer/onesFillGbatch_normalization/moving_variance_15/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_15/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_15*

index_type0
�
&batch_normalization/moving_variance_15
VariableV2*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_15*
dtype0
�
-batch_normalization/moving_variance_15/AssignAssign&batch_normalization/moving_variance_157batch_normalization/moving_variance_15/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_15*
validate_shape(
�
+batch_normalization/moving_variance_15/readIdentity&batch_normalization/moving_variance_15*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_15
�
%batch_normalization/FusedBatchNorm_15FusedBatchNormres3b_branch2a!batch_normalization/gamma_15/read batch_normalization/beta_15/read'batch_normalization/moving_mean_15/read+batch_normalization/moving_variance_15/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_15Const*
valueB
 * �:*
dtype0
F
bn3b_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3b_branch2aIdentitybn3b_branch2a*
T0
6
res3b_branch2a_reluReluscale3b_branch2a*
T0
K
res3b_branch2b/kernelConst*
valueB@�*
dtype0
�
res3b_branch2bConv2Dres3b_branch2a_relures3b_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_16/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_16*
dtype0
�
3batch_normalization/gamma_16/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_16*
dtype0
�
-batch_normalization/gamma_16/Initializer/onesFill=batch_normalization/gamma_16/Initializer/ones/shape_as_tensor3batch_normalization/gamma_16/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_16
�
batch_normalization/gamma_16
VariableV2*
dtype0*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_16
�
#batch_normalization/gamma_16/AssignAssignbatch_normalization/gamma_16-batch_normalization/gamma_16/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_16*
validate_shape(
�
!batch_normalization/gamma_16/readIdentitybatch_normalization/gamma_16*/
_class%
#!loc:@batch_normalization/gamma_16*
T0
�
=batch_normalization/beta_16/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_16*
dtype0
�
3batch_normalization/beta_16/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_16*
dtype0
�
-batch_normalization/beta_16/Initializer/zerosFill=batch_normalization/beta_16/Initializer/zeros/shape_as_tensor3batch_normalization/beta_16/Initializer/zeros/Const*

index_type0*.
_class$
" loc:@batch_normalization/beta_16*
T0
�
batch_normalization/beta_16
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_16*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_16/AssignAssignbatch_normalization/beta_16-batch_normalization/beta_16/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_16*
validate_shape(
�
 batch_normalization/beta_16/readIdentitybatch_normalization/beta_16*
T0*.
_class$
" loc:@batch_normalization/beta_16
�
Dbatch_normalization/moving_mean_16/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_16*
dtype0
�
:batch_normalization/moving_mean_16/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_16*
dtype0
�
4batch_normalization/moving_mean_16/Initializer/zerosFillDbatch_normalization/moving_mean_16/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_16/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_16
�
"batch_normalization/moving_mean_16
VariableV2*
dtype0*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_16
�
)batch_normalization/moving_mean_16/AssignAssign"batch_normalization/moving_mean_164batch_normalization/moving_mean_16/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_16*
validate_shape(
�
'batch_normalization/moving_mean_16/readIdentity"batch_normalization/moving_mean_16*5
_class+
)'loc:@batch_normalization/moving_mean_16*
T0
�
Gbatch_normalization/moving_variance_16/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_16*
dtype0
�
=batch_normalization/moving_variance_16/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_16*
dtype0
�
7batch_normalization/moving_variance_16/Initializer/onesFillGbatch_normalization/moving_variance_16/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_16/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_16
�
&batch_normalization/moving_variance_16
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_16*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_16/AssignAssign&batch_normalization/moving_variance_167batch_normalization/moving_variance_16/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_16*
validate_shape(
�
+batch_normalization/moving_variance_16/readIdentity&batch_normalization/moving_variance_16*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_16
�
%batch_normalization/FusedBatchNorm_16FusedBatchNormres3b_branch2b!batch_normalization/gamma_16/read batch_normalization/beta_16/read'batch_normalization/moving_mean_16/read+batch_normalization/moving_variance_16/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_16Const*
valueB
 * �:*
dtype0
F
bn3b_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3b_branch2bIdentitybn3b_branch2b*
T0
6
res3b_branch2b_reluReluscale3b_branch2b*
T0
K
res3b_branch2c/kernelConst*
valueB@�*
dtype0
�
res3b_branch2cConv2Dres3b_branch2b_relures3b_branch2c/kernel*
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC
�
=batch_normalization/gamma_17/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_17*
valueB:�*
dtype0
�
3batch_normalization/gamma_17/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_17*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_17/Initializer/onesFill=batch_normalization/gamma_17/Initializer/ones/shape_as_tensor3batch_normalization/gamma_17/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_17*

index_type0
�
batch_normalization/gamma_17
VariableV2*
dtype0*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_17
�
#batch_normalization/gamma_17/AssignAssignbatch_normalization/gamma_17-batch_normalization/gamma_17/Initializer/ones*/
_class%
#!loc:@batch_normalization/gamma_17*
validate_shape(*
use_locking(*
T0
�
!batch_normalization/gamma_17/readIdentitybatch_normalization/gamma_17*
T0*/
_class%
#!loc:@batch_normalization/gamma_17
�
=batch_normalization/beta_17/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_17*
valueB:�*
dtype0
�
3batch_normalization/beta_17/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_17*
valueB
 *    *
dtype0
�
-batch_normalization/beta_17/Initializer/zerosFill=batch_normalization/beta_17/Initializer/zeros/shape_as_tensor3batch_normalization/beta_17/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_17*

index_type0
�
batch_normalization/beta_17
VariableV2*.
_class$
" loc:@batch_normalization/beta_17*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_17/AssignAssignbatch_normalization/beta_17-batch_normalization/beta_17/Initializer/zeros*.
_class$
" loc:@batch_normalization/beta_17*
validate_shape(*
use_locking(*
T0
�
 batch_normalization/beta_17/readIdentitybatch_normalization/beta_17*
T0*.
_class$
" loc:@batch_normalization/beta_17
�
Dbatch_normalization/moving_mean_17/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_17*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_17/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_17*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_17/Initializer/zerosFillDbatch_normalization/moving_mean_17/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_17/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_17*

index_type0
�
"batch_normalization/moving_mean_17
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_17*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_17/AssignAssign"batch_normalization/moving_mean_174batch_normalization/moving_mean_17/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_17*
validate_shape(
�
'batch_normalization/moving_mean_17/readIdentity"batch_normalization/moving_mean_17*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_17
�
Gbatch_normalization/moving_variance_17/Initializer/ones/shape_as_tensorConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_17*
valueB:�
�
=batch_normalization/moving_variance_17/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_17*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_17/Initializer/onesFillGbatch_normalization/moving_variance_17/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_17/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_17*

index_type0
�
&batch_normalization/moving_variance_17
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_17*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_17/AssignAssign&batch_normalization/moving_variance_177batch_normalization/moving_variance_17/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_17*
validate_shape(
�
+batch_normalization/moving_variance_17/readIdentity&batch_normalization/moving_variance_17*9
_class/
-+loc:@batch_normalization/moving_variance_17*
T0
�
%batch_normalization/FusedBatchNorm_17FusedBatchNormres3b_branch2c!batch_normalization/gamma_17/read batch_normalization/beta_17/read'batch_normalization/moving_mean_17/read+batch_normalization/moving_variance_17/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_17Const*
valueB
 * �:*
dtype0
F
bn3b_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3b_branch2cIdentitybn3b_branch2c*
T0
3
res3bAdd
res3a_reluscale3b_branch2c*
T0
"

res3b_reluRelures3b*
T0
K
res3c_branch2a/kernelConst*
valueB@�*
dtype0
�
res3c_branch2aConv2D
res3b_relures3c_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_18/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_18
�
3batch_normalization/gamma_18/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_18*
dtype0
�
-batch_normalization/gamma_18/Initializer/onesFill=batch_normalization/gamma_18/Initializer/ones/shape_as_tensor3batch_normalization/gamma_18/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_18
�
batch_normalization/gamma_18
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_18*
dtype0*
	container 
�
#batch_normalization/gamma_18/AssignAssignbatch_normalization/gamma_18-batch_normalization/gamma_18/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_18*
validate_shape(
�
!batch_normalization/gamma_18/readIdentitybatch_normalization/gamma_18*
T0*/
_class%
#!loc:@batch_normalization/gamma_18
�
=batch_normalization/beta_18/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_18*
dtype0
�
3batch_normalization/beta_18/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_18*
dtype0
�
-batch_normalization/beta_18/Initializer/zerosFill=batch_normalization/beta_18/Initializer/zeros/shape_as_tensor3batch_normalization/beta_18/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_18
�
batch_normalization/beta_18
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_18
�
"batch_normalization/beta_18/AssignAssignbatch_normalization/beta_18-batch_normalization/beta_18/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_18
�
 batch_normalization/beta_18/readIdentitybatch_normalization/beta_18*
T0*.
_class$
" loc:@batch_normalization/beta_18
�
Dbatch_normalization/moving_mean_18/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_18*
dtype0
�
:batch_normalization/moving_mean_18/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_18*
dtype0
�
4batch_normalization/moving_mean_18/Initializer/zerosFillDbatch_normalization/moving_mean_18/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_18/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_18
�
"batch_normalization/moving_mean_18
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_18*
dtype0*
	container 
�
)batch_normalization/moving_mean_18/AssignAssign"batch_normalization/moving_mean_184batch_normalization/moving_mean_18/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_18
�
'batch_normalization/moving_mean_18/readIdentity"batch_normalization/moving_mean_18*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_18
�
Gbatch_normalization/moving_variance_18/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_18*
dtype0
�
=batch_normalization/moving_variance_18/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_18*
dtype0
�
7batch_normalization/moving_variance_18/Initializer/onesFillGbatch_normalization/moving_variance_18/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_18/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_18
�
&batch_normalization/moving_variance_18
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_18*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_18/AssignAssign&batch_normalization/moving_variance_187batch_normalization/moving_variance_18/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_18*
validate_shape(
�
+batch_normalization/moving_variance_18/readIdentity&batch_normalization/moving_variance_18*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_18
�
%batch_normalization/FusedBatchNorm_18FusedBatchNormres3c_branch2a!batch_normalization/gamma_18/read batch_normalization/beta_18/read'batch_normalization/moving_mean_18/read+batch_normalization/moving_variance_18/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_18Const*
valueB
 * �:*
dtype0
F
bn3c_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3c_branch2aIdentitybn3c_branch2a*
T0
6
res3c_branch2a_reluReluscale3c_branch2a*
T0
K
res3c_branch2b/kernelConst*
dtype0*
valueB@�
�
res3c_branch2bConv2Dres3c_branch2a_relures3c_branch2b/kernel*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC
�
=batch_normalization/gamma_19/Initializer/ones/shape_as_tensorConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_19*
valueB:�
�
3batch_normalization/gamma_19/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_19*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_19/Initializer/onesFill=batch_normalization/gamma_19/Initializer/ones/shape_as_tensor3batch_normalization/gamma_19/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_19*

index_type0
�
batch_normalization/gamma_19
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_19*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_19/AssignAssignbatch_normalization/gamma_19-batch_normalization/gamma_19/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_19*
validate_shape(
�
!batch_normalization/gamma_19/readIdentitybatch_normalization/gamma_19*
T0*/
_class%
#!loc:@batch_normalization/gamma_19
�
=batch_normalization/beta_19/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_19*
valueB:�*
dtype0
�
3batch_normalization/beta_19/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_19*
valueB
 *    *
dtype0
�
-batch_normalization/beta_19/Initializer/zerosFill=batch_normalization/beta_19/Initializer/zeros/shape_as_tensor3batch_normalization/beta_19/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_19*

index_type0
�
batch_normalization/beta_19
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_19*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_19/AssignAssignbatch_normalization/beta_19-batch_normalization/beta_19/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_19*
validate_shape(
�
 batch_normalization/beta_19/readIdentitybatch_normalization/beta_19*
T0*.
_class$
" loc:@batch_normalization/beta_19
�
Dbatch_normalization/moving_mean_19/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_19*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_19/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_19*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_19/Initializer/zerosFillDbatch_normalization/moving_mean_19/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_19/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_19*

index_type0
�
"batch_normalization/moving_mean_19
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_19*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_19/AssignAssign"batch_normalization/moving_mean_194batch_normalization/moving_mean_19/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_19*
validate_shape(
�
'batch_normalization/moving_mean_19/readIdentity"batch_normalization/moving_mean_19*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_19
�
Gbatch_normalization/moving_variance_19/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_19*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_19/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_19*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_19/Initializer/onesFillGbatch_normalization/moving_variance_19/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_19/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_19*

index_type0
�
&batch_normalization/moving_variance_19
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_19*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_19/AssignAssign&batch_normalization/moving_variance_197batch_normalization/moving_variance_19/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_19*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_19/readIdentity&batch_normalization/moving_variance_19*9
_class/
-+loc:@batch_normalization/moving_variance_19*
T0
�
%batch_normalization/FusedBatchNorm_19FusedBatchNormres3c_branch2b!batch_normalization/gamma_19/read batch_normalization/beta_19/read'batch_normalization/moving_mean_19/read+batch_normalization/moving_variance_19/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_19Const*
valueB
 * �:*
dtype0
F
bn3c_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3c_branch2bIdentitybn3c_branch2b*
T0
6
res3c_branch2b_reluReluscale3c_branch2b*
T0
K
res3c_branch2c/kernelConst*
valueB@�*
dtype0
�
res3c_branch2cConv2Dres3c_branch2b_relures3c_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_20/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_20*
dtype0
�
3batch_normalization/gamma_20/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_20*
dtype0
�
-batch_normalization/gamma_20/Initializer/onesFill=batch_normalization/gamma_20/Initializer/ones/shape_as_tensor3batch_normalization/gamma_20/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_20
�
batch_normalization/gamma_20
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_20*
dtype0*
	container 
�
#batch_normalization/gamma_20/AssignAssignbatch_normalization/gamma_20-batch_normalization/gamma_20/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_20*
validate_shape(
�
!batch_normalization/gamma_20/readIdentitybatch_normalization/gamma_20*
T0*/
_class%
#!loc:@batch_normalization/gamma_20
�
=batch_normalization/beta_20/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_20*
dtype0
�
3batch_normalization/beta_20/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_20
�
-batch_normalization/beta_20/Initializer/zerosFill=batch_normalization/beta_20/Initializer/zeros/shape_as_tensor3batch_normalization/beta_20/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_20
�
batch_normalization/beta_20
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_20*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_20/AssignAssignbatch_normalization/beta_20-batch_normalization/beta_20/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_20*
validate_shape(*
use_locking(
�
 batch_normalization/beta_20/readIdentitybatch_normalization/beta_20*
T0*.
_class$
" loc:@batch_normalization/beta_20
�
Dbatch_normalization/moving_mean_20/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_20*
dtype0
�
:batch_normalization/moving_mean_20/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_20*
dtype0
�
4batch_normalization/moving_mean_20/Initializer/zerosFillDbatch_normalization/moving_mean_20/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_20/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_20
�
"batch_normalization/moving_mean_20
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_20*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_20/AssignAssign"batch_normalization/moving_mean_204batch_normalization/moving_mean_20/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_20*
validate_shape(
�
'batch_normalization/moving_mean_20/readIdentity"batch_normalization/moving_mean_20*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_20
�
Gbatch_normalization/moving_variance_20/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_20*
dtype0
�
=batch_normalization/moving_variance_20/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_20*
dtype0
�
7batch_normalization/moving_variance_20/Initializer/onesFillGbatch_normalization/moving_variance_20/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_20/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_20
�
&batch_normalization/moving_variance_20
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_20*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_20/AssignAssign&batch_normalization/moving_variance_207batch_normalization/moving_variance_20/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_20*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_20/readIdentity&batch_normalization/moving_variance_20*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_20
�
%batch_normalization/FusedBatchNorm_20FusedBatchNormres3c_branch2c!batch_normalization/gamma_20/read batch_normalization/beta_20/read'batch_normalization/moving_mean_20/read+batch_normalization/moving_variance_20/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_20Const*
dtype0*
valueB
 * �:
F
bn3c_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3c_branch2cIdentitybn3c_branch2c*
T0
3
res3cAdd
res3b_reluscale3c_branch2c*
T0
"

res3c_reluRelures3c*
T0
K
res3d_branch2a/kernelConst*
valueB@�*
dtype0
�
res3d_branch2aConv2D
res3c_relures3d_branch2a/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_21/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_21*
valueB:�*
dtype0
�
3batch_normalization/gamma_21/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_21*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_21/Initializer/onesFill=batch_normalization/gamma_21/Initializer/ones/shape_as_tensor3batch_normalization/gamma_21/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_21*

index_type0
�
batch_normalization/gamma_21
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_21*
dtype0*
	container 
�
#batch_normalization/gamma_21/AssignAssignbatch_normalization/gamma_21-batch_normalization/gamma_21/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_21*
validate_shape(
�
!batch_normalization/gamma_21/readIdentitybatch_normalization/gamma_21*
T0*/
_class%
#!loc:@batch_normalization/gamma_21
�
=batch_normalization/beta_21/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_21*
valueB:�*
dtype0
�
3batch_normalization/beta_21/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_21*
valueB
 *    *
dtype0
�
-batch_normalization/beta_21/Initializer/zerosFill=batch_normalization/beta_21/Initializer/zeros/shape_as_tensor3batch_normalization/beta_21/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_21*

index_type0
�
batch_normalization/beta_21
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_21*
dtype0*
	container 
�
"batch_normalization/beta_21/AssignAssignbatch_normalization/beta_21-batch_normalization/beta_21/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_21
�
 batch_normalization/beta_21/readIdentitybatch_normalization/beta_21*.
_class$
" loc:@batch_normalization/beta_21*
T0
�
Dbatch_normalization/moving_mean_21/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_21*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_21/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_21*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_21/Initializer/zerosFillDbatch_normalization/moving_mean_21/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_21/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_21*

index_type0
�
"batch_normalization/moving_mean_21
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_21*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_21/AssignAssign"batch_normalization/moving_mean_214batch_normalization/moving_mean_21/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_21*
validate_shape(
�
'batch_normalization/moving_mean_21/readIdentity"batch_normalization/moving_mean_21*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_21
�
Gbatch_normalization/moving_variance_21/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_21*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_21/Initializer/ones/ConstConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_21*
valueB
 *  �?
�
7batch_normalization/moving_variance_21/Initializer/onesFillGbatch_normalization/moving_variance_21/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_21/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_21*

index_type0
�
&batch_normalization/moving_variance_21
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_21*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_21/AssignAssign&batch_normalization/moving_variance_217batch_normalization/moving_variance_21/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_21*
validate_shape(
�
+batch_normalization/moving_variance_21/readIdentity&batch_normalization/moving_variance_21*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_21
�
%batch_normalization/FusedBatchNorm_21FusedBatchNormres3d_branch2a!batch_normalization/gamma_21/read batch_normalization/beta_21/read'batch_normalization/moving_mean_21/read+batch_normalization/moving_variance_21/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_21Const*
valueB
 * �:*
dtype0
F
bn3d_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3d_branch2aIdentitybn3d_branch2a*
T0
6
res3d_branch2a_reluReluscale3d_branch2a*
T0
K
res3d_branch2b/kernelConst*
valueB@�*
dtype0
�
res3d_branch2bConv2Dres3d_branch2a_relures3d_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_22/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_22
�
3batch_normalization/gamma_22/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_22*
dtype0
�
-batch_normalization/gamma_22/Initializer/onesFill=batch_normalization/gamma_22/Initializer/ones/shape_as_tensor3batch_normalization/gamma_22/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_22
�
batch_normalization/gamma_22
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_22*
dtype0
�
#batch_normalization/gamma_22/AssignAssignbatch_normalization/gamma_22-batch_normalization/gamma_22/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_22*
validate_shape(
�
!batch_normalization/gamma_22/readIdentitybatch_normalization/gamma_22*
T0*/
_class%
#!loc:@batch_normalization/gamma_22
�
=batch_normalization/beta_22/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:�*.
_class$
" loc:@batch_normalization/beta_22
�
3batch_normalization/beta_22/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_22*
dtype0
�
-batch_normalization/beta_22/Initializer/zerosFill=batch_normalization/beta_22/Initializer/zeros/shape_as_tensor3batch_normalization/beta_22/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_22
�
batch_normalization/beta_22
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_22*
dtype0*
	container 
�
"batch_normalization/beta_22/AssignAssignbatch_normalization/beta_22-batch_normalization/beta_22/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_22*
validate_shape(*
use_locking(
�
 batch_normalization/beta_22/readIdentitybatch_normalization/beta_22*
T0*.
_class$
" loc:@batch_normalization/beta_22
�
Dbatch_normalization/moving_mean_22/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_22*
dtype0
�
:batch_normalization/moving_mean_22/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_22*
dtype0
�
4batch_normalization/moving_mean_22/Initializer/zerosFillDbatch_normalization/moving_mean_22/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_22/Initializer/zeros/Const*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_22*
T0
�
"batch_normalization/moving_mean_22
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_22*
dtype0*
	container 
�
)batch_normalization/moving_mean_22/AssignAssign"batch_normalization/moving_mean_224batch_normalization/moving_mean_22/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_22*
validate_shape(
�
'batch_normalization/moving_mean_22/readIdentity"batch_normalization/moving_mean_22*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_22
�
Gbatch_normalization/moving_variance_22/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_22
�
=batch_normalization/moving_variance_22/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_22*
dtype0
�
7batch_normalization/moving_variance_22/Initializer/onesFillGbatch_normalization/moving_variance_22/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_22/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_22
�
&batch_normalization/moving_variance_22
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_22*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_22/AssignAssign&batch_normalization/moving_variance_227batch_normalization/moving_variance_22/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_22*
validate_shape(
�
+batch_normalization/moving_variance_22/readIdentity&batch_normalization/moving_variance_22*9
_class/
-+loc:@batch_normalization/moving_variance_22*
T0
�
%batch_normalization/FusedBatchNorm_22FusedBatchNormres3d_branch2b!batch_normalization/gamma_22/read batch_normalization/beta_22/read'batch_normalization/moving_mean_22/read+batch_normalization/moving_variance_22/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_22Const*
valueB
 * �:*
dtype0
F
bn3d_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3d_branch2bIdentitybn3d_branch2b*
T0
6
res3d_branch2b_reluReluscale3d_branch2b*
T0
K
res3d_branch2c/kernelConst*
valueB@�*
dtype0
�
res3d_branch2cConv2Dres3d_branch2b_relures3d_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_23/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_23*
valueB:�*
dtype0
�
3batch_normalization/gamma_23/Initializer/ones/ConstConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_23*
valueB
 *  �?
�
-batch_normalization/gamma_23/Initializer/onesFill=batch_normalization/gamma_23/Initializer/ones/shape_as_tensor3batch_normalization/gamma_23/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_23*

index_type0
�
batch_normalization/gamma_23
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_23*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_23/AssignAssignbatch_normalization/gamma_23-batch_normalization/gamma_23/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_23*
validate_shape(
�
!batch_normalization/gamma_23/readIdentitybatch_normalization/gamma_23*
T0*/
_class%
#!loc:@batch_normalization/gamma_23
�
=batch_normalization/beta_23/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_23*
valueB:�*
dtype0
�
3batch_normalization/beta_23/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_23*
valueB
 *    *
dtype0
�
-batch_normalization/beta_23/Initializer/zerosFill=batch_normalization/beta_23/Initializer/zeros/shape_as_tensor3batch_normalization/beta_23/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_23*

index_type0
�
batch_normalization/beta_23
VariableV2*.
_class$
" loc:@batch_normalization/beta_23*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_23/AssignAssignbatch_normalization/beta_23-batch_normalization/beta_23/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_23*
validate_shape(
�
 batch_normalization/beta_23/readIdentitybatch_normalization/beta_23*
T0*.
_class$
" loc:@batch_normalization/beta_23
�
Dbatch_normalization/moving_mean_23/Initializer/zeros/shape_as_tensorConst*
dtype0*5
_class+
)'loc:@batch_normalization/moving_mean_23*
valueB:�
�
:batch_normalization/moving_mean_23/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_23*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_23/Initializer/zerosFillDbatch_normalization/moving_mean_23/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_23/Initializer/zeros/Const*5
_class+
)'loc:@batch_normalization/moving_mean_23*

index_type0*
T0
�
"batch_normalization/moving_mean_23
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_23*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_23/AssignAssign"batch_normalization/moving_mean_234batch_normalization/moving_mean_23/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_23*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_23/readIdentity"batch_normalization/moving_mean_23*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_23
�
Gbatch_normalization/moving_variance_23/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_23*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_23/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_23*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_23/Initializer/onesFillGbatch_normalization/moving_variance_23/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_23/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_23*

index_type0
�
&batch_normalization/moving_variance_23
VariableV2*
dtype0*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_23
�
-batch_normalization/moving_variance_23/AssignAssign&batch_normalization/moving_variance_237batch_normalization/moving_variance_23/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_23*
validate_shape(
�
+batch_normalization/moving_variance_23/readIdentity&batch_normalization/moving_variance_23*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_23
�
%batch_normalization/FusedBatchNorm_23FusedBatchNormres3d_branch2c!batch_normalization/gamma_23/read batch_normalization/beta_23/read'batch_normalization/moving_mean_23/read+batch_normalization/moving_variance_23/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_23Const*
dtype0*
valueB
 * �:
F
bn3d_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale3d_branch2cIdentitybn3d_branch2c*
T0
3
res3dAdd
res3c_reluscale3d_branch2c*
T0
"

res3d_reluRelures3d*
T0
J
res4a_branch1/kernelConst*
valueB@�*
dtype0
�
res4a_branch1Conv2D
res3d_relures4a_branch1/kernel*
paddingVALID*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_24/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_24*
dtype0
�
3batch_normalization/gamma_24/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_24*
dtype0
�
-batch_normalization/gamma_24/Initializer/onesFill=batch_normalization/gamma_24/Initializer/ones/shape_as_tensor3batch_normalization/gamma_24/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_24
�
batch_normalization/gamma_24
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_24*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_24/AssignAssignbatch_normalization/gamma_24-batch_normalization/gamma_24/Initializer/ones*
T0*/
_class%
#!loc:@batch_normalization/gamma_24*
validate_shape(*
use_locking(
�
!batch_normalization/gamma_24/readIdentitybatch_normalization/gamma_24*
T0*/
_class%
#!loc:@batch_normalization/gamma_24
�
=batch_normalization/beta_24/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_24*
dtype0
�
3batch_normalization/beta_24/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_24*
dtype0
�
-batch_normalization/beta_24/Initializer/zerosFill=batch_normalization/beta_24/Initializer/zeros/shape_as_tensor3batch_normalization/beta_24/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_24
�
batch_normalization/beta_24
VariableV2*.
_class$
" loc:@batch_normalization/beta_24*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_24/AssignAssignbatch_normalization/beta_24-batch_normalization/beta_24/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_24*
validate_shape(
�
 batch_normalization/beta_24/readIdentitybatch_normalization/beta_24*
T0*.
_class$
" loc:@batch_normalization/beta_24
�
Dbatch_normalization/moving_mean_24/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_24
�
:batch_normalization/moving_mean_24/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_24*
dtype0
�
4batch_normalization/moving_mean_24/Initializer/zerosFillDbatch_normalization/moving_mean_24/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_24/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_24
�
"batch_normalization/moving_mean_24
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_24*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_24/AssignAssign"batch_normalization/moving_mean_244batch_normalization/moving_mean_24/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_24*
validate_shape(
�
'batch_normalization/moving_mean_24/readIdentity"batch_normalization/moving_mean_24*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_24
�
Gbatch_normalization/moving_variance_24/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_24*
dtype0
�
=batch_normalization/moving_variance_24/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_24*
dtype0
�
7batch_normalization/moving_variance_24/Initializer/onesFillGbatch_normalization/moving_variance_24/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_24/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_24
�
&batch_normalization/moving_variance_24
VariableV2*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_24*
dtype0
�
-batch_normalization/moving_variance_24/AssignAssign&batch_normalization/moving_variance_247batch_normalization/moving_variance_24/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_24*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_24/readIdentity&batch_normalization/moving_variance_24*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_24
�
%batch_normalization/FusedBatchNorm_24FusedBatchNormres4a_branch1!batch_normalization/gamma_24/read batch_normalization/beta_24/read'batch_normalization/moving_mean_24/read+batch_normalization/moving_variance_24/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_24Const*
valueB
 * �:*
dtype0
E
bn4a_branch1Identity"batch_normalization/FusedBatchNorm*
T0
2
scale4a_branch1Identitybn4a_branch1*
T0
K
res4a_branch2a/kernelConst*
valueB@�*
dtype0
�
res4a_branch2aConv2D
res3d_relures4a_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_25/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_25*
valueB:�*
dtype0
�
3batch_normalization/gamma_25/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_25*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_25/Initializer/onesFill=batch_normalization/gamma_25/Initializer/ones/shape_as_tensor3batch_normalization/gamma_25/Initializer/ones/Const*/
_class%
#!loc:@batch_normalization/gamma_25*

index_type0*
T0
�
batch_normalization/gamma_25
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_25*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_25/AssignAssignbatch_normalization/gamma_25-batch_normalization/gamma_25/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_25*
validate_shape(
�
!batch_normalization/gamma_25/readIdentitybatch_normalization/gamma_25*
T0*/
_class%
#!loc:@batch_normalization/gamma_25
�
=batch_normalization/beta_25/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_25*
valueB:�*
dtype0
�
3batch_normalization/beta_25/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_25*
valueB
 *    *
dtype0
�
-batch_normalization/beta_25/Initializer/zerosFill=batch_normalization/beta_25/Initializer/zeros/shape_as_tensor3batch_normalization/beta_25/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_25*

index_type0
�
batch_normalization/beta_25
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_25*
dtype0*
	container 
�
"batch_normalization/beta_25/AssignAssignbatch_normalization/beta_25-batch_normalization/beta_25/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_25*
validate_shape(
�
 batch_normalization/beta_25/readIdentitybatch_normalization/beta_25*
T0*.
_class$
" loc:@batch_normalization/beta_25
�
Dbatch_normalization/moving_mean_25/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_25*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_25/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_25*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_25/Initializer/zerosFillDbatch_normalization/moving_mean_25/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_25/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_25*

index_type0
�
"batch_normalization/moving_mean_25
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_25*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_25/AssignAssign"batch_normalization/moving_mean_254batch_normalization/moving_mean_25/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_25
�
'batch_normalization/moving_mean_25/readIdentity"batch_normalization/moving_mean_25*5
_class+
)'loc:@batch_normalization/moving_mean_25*
T0
�
Gbatch_normalization/moving_variance_25/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_25*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_25/Initializer/ones/ConstConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_25*
valueB
 *  �?
�
7batch_normalization/moving_variance_25/Initializer/onesFillGbatch_normalization/moving_variance_25/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_25/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_25*

index_type0
�
&batch_normalization/moving_variance_25
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_25*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_25/AssignAssign&batch_normalization/moving_variance_257batch_normalization/moving_variance_25/Initializer/ones*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_25
�
+batch_normalization/moving_variance_25/readIdentity&batch_normalization/moving_variance_25*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_25
�
%batch_normalization/FusedBatchNorm_25FusedBatchNormres4a_branch2a!batch_normalization/gamma_25/read batch_normalization/beta_25/read'batch_normalization/moving_mean_25/read+batch_normalization/moving_variance_25/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_25Const*
dtype0*
valueB
 * �:
F
bn4a_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4a_branch2aIdentitybn4a_branch2a*
T0
6
res4a_branch2a_reluReluscale4a_branch2a*
T0
K
res4a_branch2b/kernelConst*
valueB@�*
dtype0
�
res4a_branch2bConv2Dres4a_branch2a_relures4a_branch2b/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_26/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_26*
dtype0
�
3batch_normalization/gamma_26/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_26*
dtype0
�
-batch_normalization/gamma_26/Initializer/onesFill=batch_normalization/gamma_26/Initializer/ones/shape_as_tensor3batch_normalization/gamma_26/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_26
�
batch_normalization/gamma_26
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_26*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_26/AssignAssignbatch_normalization/gamma_26-batch_normalization/gamma_26/Initializer/ones*/
_class%
#!loc:@batch_normalization/gamma_26*
validate_shape(*
use_locking(*
T0
�
!batch_normalization/gamma_26/readIdentitybatch_normalization/gamma_26*
T0*/
_class%
#!loc:@batch_normalization/gamma_26
�
=batch_normalization/beta_26/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:�*.
_class$
" loc:@batch_normalization/beta_26
�
3batch_normalization/beta_26/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_26*
dtype0
�
-batch_normalization/beta_26/Initializer/zerosFill=batch_normalization/beta_26/Initializer/zeros/shape_as_tensor3batch_normalization/beta_26/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_26
�
batch_normalization/beta_26
VariableV2*.
_class$
" loc:@batch_normalization/beta_26*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_26/AssignAssignbatch_normalization/beta_26-batch_normalization/beta_26/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_26*
validate_shape(
�
 batch_normalization/beta_26/readIdentitybatch_normalization/beta_26*
T0*.
_class$
" loc:@batch_normalization/beta_26
�
Dbatch_normalization/moving_mean_26/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_26*
dtype0
�
:batch_normalization/moving_mean_26/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_26
�
4batch_normalization/moving_mean_26/Initializer/zerosFillDbatch_normalization/moving_mean_26/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_26/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_26
�
"batch_normalization/moving_mean_26
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_26*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_26/AssignAssign"batch_normalization/moving_mean_264batch_normalization/moving_mean_26/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_26*
validate_shape(
�
'batch_normalization/moving_mean_26/readIdentity"batch_normalization/moving_mean_26*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_26
�
Gbatch_normalization/moving_variance_26/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_26*
dtype0
�
=batch_normalization/moving_variance_26/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_26*
dtype0
�
7batch_normalization/moving_variance_26/Initializer/onesFillGbatch_normalization/moving_variance_26/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_26/Initializer/ones/Const*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_26*
T0
�
&batch_normalization/moving_variance_26
VariableV2*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_26*
dtype0
�
-batch_normalization/moving_variance_26/AssignAssign&batch_normalization/moving_variance_267batch_normalization/moving_variance_26/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_26*
validate_shape(
�
+batch_normalization/moving_variance_26/readIdentity&batch_normalization/moving_variance_26*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_26
�
%batch_normalization/FusedBatchNorm_26FusedBatchNormres4a_branch2b!batch_normalization/gamma_26/read batch_normalization/beta_26/read'batch_normalization/moving_mean_26/read+batch_normalization/moving_variance_26/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_26Const*
valueB
 * �:*
dtype0
F
bn4a_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4a_branch2bIdentitybn4a_branch2b*
T0
6
res4a_branch2b_reluReluscale4a_branch2b*
T0
K
res4a_branch2c/kernelConst*
valueB@�*
dtype0
�
res4a_branch2cConv2Dres4a_branch2b_relures4a_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_27/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_27*
valueB:�*
dtype0
�
3batch_normalization/gamma_27/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_27*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_27/Initializer/onesFill=batch_normalization/gamma_27/Initializer/ones/shape_as_tensor3batch_normalization/gamma_27/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_27*

index_type0
�
batch_normalization/gamma_27
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_27*
dtype0*
	container 
�
#batch_normalization/gamma_27/AssignAssignbatch_normalization/gamma_27-batch_normalization/gamma_27/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_27*
validate_shape(
�
!batch_normalization/gamma_27/readIdentitybatch_normalization/gamma_27*
T0*/
_class%
#!loc:@batch_normalization/gamma_27
�
=batch_normalization/beta_27/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_27*
valueB:�*
dtype0
�
3batch_normalization/beta_27/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_27*
valueB
 *    *
dtype0
�
-batch_normalization/beta_27/Initializer/zerosFill=batch_normalization/beta_27/Initializer/zeros/shape_as_tensor3batch_normalization/beta_27/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_27*

index_type0
�
batch_normalization/beta_27
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_27*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_27/AssignAssignbatch_normalization/beta_27-batch_normalization/beta_27/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_27*
validate_shape(
�
 batch_normalization/beta_27/readIdentitybatch_normalization/beta_27*
T0*.
_class$
" loc:@batch_normalization/beta_27
�
Dbatch_normalization/moving_mean_27/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_27*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_27/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_27*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_27/Initializer/zerosFillDbatch_normalization/moving_mean_27/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_27/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_27*

index_type0
�
"batch_normalization/moving_mean_27
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_27*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_27/AssignAssign"batch_normalization/moving_mean_274batch_normalization/moving_mean_27/Initializer/zeros*5
_class+
)'loc:@batch_normalization/moving_mean_27*
validate_shape(*
use_locking(*
T0
�
'batch_normalization/moving_mean_27/readIdentity"batch_normalization/moving_mean_27*5
_class+
)'loc:@batch_normalization/moving_mean_27*
T0
�
Gbatch_normalization/moving_variance_27/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_27*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_27/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_27*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_27/Initializer/onesFillGbatch_normalization/moving_variance_27/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_27/Initializer/ones/Const*9
_class/
-+loc:@batch_normalization/moving_variance_27*

index_type0*
T0
�
&batch_normalization/moving_variance_27
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_27*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_27/AssignAssign&batch_normalization/moving_variance_277batch_normalization/moving_variance_27/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_27*
validate_shape(
�
+batch_normalization/moving_variance_27/readIdentity&batch_normalization/moving_variance_27*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_27
�
%batch_normalization/FusedBatchNorm_27FusedBatchNormres4a_branch2c!batch_normalization/gamma_27/read batch_normalization/beta_27/read'batch_normalization/moving_mean_27/read+batch_normalization/moving_variance_27/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_27Const*
dtype0*
valueB
 * �:
F
bn4a_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4a_branch2cIdentitybn4a_branch2c*
T0
8
res4aAddscale4a_branch1scale4a_branch2c*
T0
"

res4a_reluRelures4a*
T0
K
res4b_branch2a/kernelConst*
valueB@�*
dtype0
�
res4b_branch2aConv2D
res4a_relures4b_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_28/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_28*
dtype0
�
3batch_normalization/gamma_28/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_28*
dtype0
�
-batch_normalization/gamma_28/Initializer/onesFill=batch_normalization/gamma_28/Initializer/ones/shape_as_tensor3batch_normalization/gamma_28/Initializer/ones/Const*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_28*
T0
�
batch_normalization/gamma_28
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_28*
dtype0*
	container 
�
#batch_normalization/gamma_28/AssignAssignbatch_normalization/gamma_28-batch_normalization/gamma_28/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_28*
validate_shape(
�
!batch_normalization/gamma_28/readIdentitybatch_normalization/gamma_28*
T0*/
_class%
#!loc:@batch_normalization/gamma_28
�
=batch_normalization/beta_28/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_28*
dtype0
�
3batch_normalization/beta_28/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_28*
dtype0
�
-batch_normalization/beta_28/Initializer/zerosFill=batch_normalization/beta_28/Initializer/zeros/shape_as_tensor3batch_normalization/beta_28/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_28
�
batch_normalization/beta_28
VariableV2*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_28*
dtype0
�
"batch_normalization/beta_28/AssignAssignbatch_normalization/beta_28-batch_normalization/beta_28/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_28*
validate_shape(
�
 batch_normalization/beta_28/readIdentitybatch_normalization/beta_28*
T0*.
_class$
" loc:@batch_normalization/beta_28
�
Dbatch_normalization/moving_mean_28/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_28*
dtype0
�
:batch_normalization/moving_mean_28/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_28*
dtype0
�
4batch_normalization/moving_mean_28/Initializer/zerosFillDbatch_normalization/moving_mean_28/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_28/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_28
�
"batch_normalization/moving_mean_28
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_28*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_28/AssignAssign"batch_normalization/moving_mean_284batch_normalization/moving_mean_28/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_28
�
'batch_normalization/moving_mean_28/readIdentity"batch_normalization/moving_mean_28*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_28
�
Gbatch_normalization/moving_variance_28/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_28*
dtype0
�
=batch_normalization/moving_variance_28/Initializer/ones/ConstConst*
dtype0*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_28
�
7batch_normalization/moving_variance_28/Initializer/onesFillGbatch_normalization/moving_variance_28/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_28/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_28
�
&batch_normalization/moving_variance_28
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_28*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_28/AssignAssign&batch_normalization/moving_variance_287batch_normalization/moving_variance_28/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_28*
validate_shape(
�
+batch_normalization/moving_variance_28/readIdentity&batch_normalization/moving_variance_28*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_28
�
%batch_normalization/FusedBatchNorm_28FusedBatchNormres4b_branch2a!batch_normalization/gamma_28/read batch_normalization/beta_28/read'batch_normalization/moving_mean_28/read+batch_normalization/moving_variance_28/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_28Const*
valueB
 * �:*
dtype0
F
bn4b_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4b_branch2aIdentitybn4b_branch2a*
T0
6
res4b_branch2a_reluReluscale4b_branch2a*
T0
K
res4b_branch2b/kernelConst*
valueB@�*
dtype0
�
res4b_branch2bConv2Dres4b_branch2a_relures4b_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_29/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_29*
valueB:�*
dtype0
�
3batch_normalization/gamma_29/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_29*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_29/Initializer/onesFill=batch_normalization/gamma_29/Initializer/ones/shape_as_tensor3batch_normalization/gamma_29/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_29*

index_type0
�
batch_normalization/gamma_29
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_29*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_29/AssignAssignbatch_normalization/gamma_29-batch_normalization/gamma_29/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_29*
validate_shape(
�
!batch_normalization/gamma_29/readIdentitybatch_normalization/gamma_29*
T0*/
_class%
#!loc:@batch_normalization/gamma_29
�
=batch_normalization/beta_29/Initializer/zeros/shape_as_tensorConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_29*
valueB:�
�
3batch_normalization/beta_29/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_29*
valueB
 *    *
dtype0
�
-batch_normalization/beta_29/Initializer/zerosFill=batch_normalization/beta_29/Initializer/zeros/shape_as_tensor3batch_normalization/beta_29/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_29*

index_type0
�
batch_normalization/beta_29
VariableV2*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_29*
dtype0
�
"batch_normalization/beta_29/AssignAssignbatch_normalization/beta_29-batch_normalization/beta_29/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_29*
validate_shape(
�
 batch_normalization/beta_29/readIdentitybatch_normalization/beta_29*
T0*.
_class$
" loc:@batch_normalization/beta_29
�
Dbatch_normalization/moving_mean_29/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_29*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_29/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_29*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_29/Initializer/zerosFillDbatch_normalization/moving_mean_29/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_29/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_29*

index_type0
�
"batch_normalization/moving_mean_29
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_29*
dtype0*
	container 
�
)batch_normalization/moving_mean_29/AssignAssign"batch_normalization/moving_mean_294batch_normalization/moving_mean_29/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_29
�
'batch_normalization/moving_mean_29/readIdentity"batch_normalization/moving_mean_29*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_29
�
Gbatch_normalization/moving_variance_29/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_29*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_29/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_29*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_29/Initializer/onesFillGbatch_normalization/moving_variance_29/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_29/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_29*

index_type0
�
&batch_normalization/moving_variance_29
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_29*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_29/AssignAssign&batch_normalization/moving_variance_297batch_normalization/moving_variance_29/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_29*
validate_shape(
�
+batch_normalization/moving_variance_29/readIdentity&batch_normalization/moving_variance_29*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_29
�
%batch_normalization/FusedBatchNorm_29FusedBatchNormres4b_branch2b!batch_normalization/gamma_29/read batch_normalization/beta_29/read'batch_normalization/moving_mean_29/read+batch_normalization/moving_variance_29/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
I
batch_normalization/Const_29Const*
valueB
 * �:*
dtype0
F
bn4b_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4b_branch2bIdentitybn4b_branch2b*
T0
6
res4b_branch2b_reluReluscale4b_branch2b*
T0
K
res4b_branch2c/kernelConst*
valueB@�*
dtype0
�
res4b_branch2cConv2Dres4b_branch2b_relures4b_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_30/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_30*
dtype0
�
3batch_normalization/gamma_30/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_30*
dtype0
�
-batch_normalization/gamma_30/Initializer/onesFill=batch_normalization/gamma_30/Initializer/ones/shape_as_tensor3batch_normalization/gamma_30/Initializer/ones/Const*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_30*
T0
�
batch_normalization/gamma_30
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_30*
dtype0*
	container 
�
#batch_normalization/gamma_30/AssignAssignbatch_normalization/gamma_30-batch_normalization/gamma_30/Initializer/ones*
T0*/
_class%
#!loc:@batch_normalization/gamma_30*
validate_shape(*
use_locking(
�
!batch_normalization/gamma_30/readIdentitybatch_normalization/gamma_30*
T0*/
_class%
#!loc:@batch_normalization/gamma_30
�
=batch_normalization/beta_30/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_30*
dtype0
�
3batch_normalization/beta_30/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_30*
dtype0
�
-batch_normalization/beta_30/Initializer/zerosFill=batch_normalization/beta_30/Initializer/zeros/shape_as_tensor3batch_normalization/beta_30/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_30
�
batch_normalization/beta_30
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_30*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_30/AssignAssignbatch_normalization/beta_30-batch_normalization/beta_30/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_30*
validate_shape(
�
 batch_normalization/beta_30/readIdentitybatch_normalization/beta_30*
T0*.
_class$
" loc:@batch_normalization/beta_30
�
Dbatch_normalization/moving_mean_30/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_30*
dtype0
�
:batch_normalization/moving_mean_30/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_30*
dtype0
�
4batch_normalization/moving_mean_30/Initializer/zerosFillDbatch_normalization/moving_mean_30/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_30/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_30
�
"batch_normalization/moving_mean_30
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_30*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_30/AssignAssign"batch_normalization/moving_mean_304batch_normalization/moving_mean_30/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_30*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_30/readIdentity"batch_normalization/moving_mean_30*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_30
�
Gbatch_normalization/moving_variance_30/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_30*
dtype0
�
=batch_normalization/moving_variance_30/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_30*
dtype0
�
7batch_normalization/moving_variance_30/Initializer/onesFillGbatch_normalization/moving_variance_30/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_30/Initializer/ones/Const*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_30*
T0
�
&batch_normalization/moving_variance_30
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_30*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_30/AssignAssign&batch_normalization/moving_variance_307batch_normalization/moving_variance_30/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_30*
validate_shape(
�
+batch_normalization/moving_variance_30/readIdentity&batch_normalization/moving_variance_30*9
_class/
-+loc:@batch_normalization/moving_variance_30*
T0
�
%batch_normalization/FusedBatchNorm_30FusedBatchNormres4b_branch2c!batch_normalization/gamma_30/read batch_normalization/beta_30/read'batch_normalization/moving_mean_30/read+batch_normalization/moving_variance_30/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_30Const*
dtype0*
valueB
 * �:
F
bn4b_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4b_branch2cIdentitybn4b_branch2c*
T0
3
res4bAdd
res4a_reluscale4b_branch2c*
T0
"

res4b_reluRelures4b*
T0
K
res4c_branch2a/kernelConst*
valueB@�*
dtype0
�
res4c_branch2aConv2D
res4b_relures4c_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_31/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_31*
valueB:�*
dtype0
�
3batch_normalization/gamma_31/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_31*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_31/Initializer/onesFill=batch_normalization/gamma_31/Initializer/ones/shape_as_tensor3batch_normalization/gamma_31/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_31*

index_type0
�
batch_normalization/gamma_31
VariableV2*
dtype0*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_31
�
#batch_normalization/gamma_31/AssignAssignbatch_normalization/gamma_31-batch_normalization/gamma_31/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_31*
validate_shape(
�
!batch_normalization/gamma_31/readIdentitybatch_normalization/gamma_31*
T0*/
_class%
#!loc:@batch_normalization/gamma_31
�
=batch_normalization/beta_31/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_31*
valueB:�*
dtype0
�
3batch_normalization/beta_31/Initializer/zeros/ConstConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_31*
valueB
 *    
�
-batch_normalization/beta_31/Initializer/zerosFill=batch_normalization/beta_31/Initializer/zeros/shape_as_tensor3batch_normalization/beta_31/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_31*

index_type0
�
batch_normalization/beta_31
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_31
�
"batch_normalization/beta_31/AssignAssignbatch_normalization/beta_31-batch_normalization/beta_31/Initializer/zeros*.
_class$
" loc:@batch_normalization/beta_31*
validate_shape(*
use_locking(*
T0
�
 batch_normalization/beta_31/readIdentitybatch_normalization/beta_31*
T0*.
_class$
" loc:@batch_normalization/beta_31
�
Dbatch_normalization/moving_mean_31/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_31*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_31/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_31*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_31/Initializer/zerosFillDbatch_normalization/moving_mean_31/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_31/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_31*

index_type0
�
"batch_normalization/moving_mean_31
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_31*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_31/AssignAssign"batch_normalization/moving_mean_314batch_normalization/moving_mean_31/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_31*
validate_shape(
�
'batch_normalization/moving_mean_31/readIdentity"batch_normalization/moving_mean_31*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_31
�
Gbatch_normalization/moving_variance_31/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_31*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_31/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_31*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_31/Initializer/onesFillGbatch_normalization/moving_variance_31/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_31/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_31*

index_type0
�
&batch_normalization/moving_variance_31
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_31*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_31/AssignAssign&batch_normalization/moving_variance_317batch_normalization/moving_variance_31/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_31*
validate_shape(
�
+batch_normalization/moving_variance_31/readIdentity&batch_normalization/moving_variance_31*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_31
�
%batch_normalization/FusedBatchNorm_31FusedBatchNormres4c_branch2a!batch_normalization/gamma_31/read batch_normalization/beta_31/read'batch_normalization/moving_mean_31/read+batch_normalization/moving_variance_31/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
I
batch_normalization/Const_31Const*
valueB
 * �:*
dtype0
F
bn4c_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4c_branch2aIdentitybn4c_branch2a*
T0
6
res4c_branch2a_reluReluscale4c_branch2a*
T0
K
res4c_branch2b/kernelConst*
valueB@�*
dtype0
�
res4c_branch2bConv2Dres4c_branch2a_relures4c_branch2b/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_32/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_32*
dtype0
�
3batch_normalization/gamma_32/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_32*
dtype0
�
-batch_normalization/gamma_32/Initializer/onesFill=batch_normalization/gamma_32/Initializer/ones/shape_as_tensor3batch_normalization/gamma_32/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_32
�
batch_normalization/gamma_32
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_32*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_32/AssignAssignbatch_normalization/gamma_32-batch_normalization/gamma_32/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_32*
validate_shape(
�
!batch_normalization/gamma_32/readIdentitybatch_normalization/gamma_32*
T0*/
_class%
#!loc:@batch_normalization/gamma_32
�
=batch_normalization/beta_32/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:�*.
_class$
" loc:@batch_normalization/beta_32
�
3batch_normalization/beta_32/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_32*
dtype0
�
-batch_normalization/beta_32/Initializer/zerosFill=batch_normalization/beta_32/Initializer/zeros/shape_as_tensor3batch_normalization/beta_32/Initializer/zeros/Const*

index_type0*.
_class$
" loc:@batch_normalization/beta_32*
T0
�
batch_normalization/beta_32
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_32*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_32/AssignAssignbatch_normalization/beta_32-batch_normalization/beta_32/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_32*
validate_shape(
�
 batch_normalization/beta_32/readIdentitybatch_normalization/beta_32*
T0*.
_class$
" loc:@batch_normalization/beta_32
�
Dbatch_normalization/moving_mean_32/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_32*
dtype0
�
:batch_normalization/moving_mean_32/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_32*
dtype0
�
4batch_normalization/moving_mean_32/Initializer/zerosFillDbatch_normalization/moving_mean_32/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_32/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_32
�
"batch_normalization/moving_mean_32
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_32*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_32/AssignAssign"batch_normalization/moving_mean_324batch_normalization/moving_mean_32/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_32*
validate_shape(
�
'batch_normalization/moving_mean_32/readIdentity"batch_normalization/moving_mean_32*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_32
�
Gbatch_normalization/moving_variance_32/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_32*
dtype0
�
=batch_normalization/moving_variance_32/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_32*
dtype0
�
7batch_normalization/moving_variance_32/Initializer/onesFillGbatch_normalization/moving_variance_32/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_32/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_32
�
&batch_normalization/moving_variance_32
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_32*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_32/AssignAssign&batch_normalization/moving_variance_327batch_normalization/moving_variance_32/Initializer/ones*9
_class/
-+loc:@batch_normalization/moving_variance_32*
validate_shape(*
use_locking(*
T0
�
+batch_normalization/moving_variance_32/readIdentity&batch_normalization/moving_variance_32*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_32
�
%batch_normalization/FusedBatchNorm_32FusedBatchNormres4c_branch2b!batch_normalization/gamma_32/read batch_normalization/beta_32/read'batch_normalization/moving_mean_32/read+batch_normalization/moving_variance_32/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_32Const*
valueB
 * �:*
dtype0
F
bn4c_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4c_branch2bIdentitybn4c_branch2b*
T0
6
res4c_branch2b_reluReluscale4c_branch2b*
T0
K
res4c_branch2c/kernelConst*
valueB@�*
dtype0
�
res4c_branch2cConv2Dres4c_branch2b_relures4c_branch2c/kernel*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_33/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_33*
valueB:�*
dtype0
�
3batch_normalization/gamma_33/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_33*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_33/Initializer/onesFill=batch_normalization/gamma_33/Initializer/ones/shape_as_tensor3batch_normalization/gamma_33/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_33*

index_type0
�
batch_normalization/gamma_33
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_33*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_33/AssignAssignbatch_normalization/gamma_33-batch_normalization/gamma_33/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_33*
validate_shape(
�
!batch_normalization/gamma_33/readIdentitybatch_normalization/gamma_33*
T0*/
_class%
#!loc:@batch_normalization/gamma_33
�
=batch_normalization/beta_33/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_33*
valueB:�*
dtype0
�
3batch_normalization/beta_33/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_33*
valueB
 *    *
dtype0
�
-batch_normalization/beta_33/Initializer/zerosFill=batch_normalization/beta_33/Initializer/zeros/shape_as_tensor3batch_normalization/beta_33/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_33*

index_type0
�
batch_normalization/beta_33
VariableV2*.
_class$
" loc:@batch_normalization/beta_33*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_33/AssignAssignbatch_normalization/beta_33-batch_normalization/beta_33/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_33*
validate_shape(
�
 batch_normalization/beta_33/readIdentitybatch_normalization/beta_33*
T0*.
_class$
" loc:@batch_normalization/beta_33
�
Dbatch_normalization/moving_mean_33/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_33*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_33/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_33*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_33/Initializer/zerosFillDbatch_normalization/moving_mean_33/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_33/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_33*

index_type0
�
"batch_normalization/moving_mean_33
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_33*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_33/AssignAssign"batch_normalization/moving_mean_334batch_normalization/moving_mean_33/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_33*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_33/readIdentity"batch_normalization/moving_mean_33*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_33
�
Gbatch_normalization/moving_variance_33/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_33*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_33/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_33*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_33/Initializer/onesFillGbatch_normalization/moving_variance_33/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_33/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_33*

index_type0
�
&batch_normalization/moving_variance_33
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_33*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_33/AssignAssign&batch_normalization/moving_variance_337batch_normalization/moving_variance_33/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_33*
validate_shape(
�
+batch_normalization/moving_variance_33/readIdentity&batch_normalization/moving_variance_33*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_33
�
%batch_normalization/FusedBatchNorm_33FusedBatchNormres4c_branch2c!batch_normalization/gamma_33/read batch_normalization/beta_33/read'batch_normalization/moving_mean_33/read+batch_normalization/moving_variance_33/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_33Const*
valueB
 * �:*
dtype0
F
bn4c_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4c_branch2cIdentitybn4c_branch2c*
T0
3
res4cAdd
res4b_reluscale4c_branch2c*
T0
"

res4c_reluRelures4c*
T0
K
res4d_branch2a/kernelConst*
valueB@�*
dtype0
�
res4d_branch2aConv2D
res4c_relures4d_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_34/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_34*
dtype0
�
3batch_normalization/gamma_34/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_34*
dtype0
�
-batch_normalization/gamma_34/Initializer/onesFill=batch_normalization/gamma_34/Initializer/ones/shape_as_tensor3batch_normalization/gamma_34/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_34
�
batch_normalization/gamma_34
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_34*
dtype0*
	container 
�
#batch_normalization/gamma_34/AssignAssignbatch_normalization/gamma_34-batch_normalization/gamma_34/Initializer/ones*
T0*/
_class%
#!loc:@batch_normalization/gamma_34*
validate_shape(*
use_locking(
�
!batch_normalization/gamma_34/readIdentitybatch_normalization/gamma_34*
T0*/
_class%
#!loc:@batch_normalization/gamma_34
�
=batch_normalization/beta_34/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_34*
dtype0
�
3batch_normalization/beta_34/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_34*
dtype0
�
-batch_normalization/beta_34/Initializer/zerosFill=batch_normalization/beta_34/Initializer/zeros/shape_as_tensor3batch_normalization/beta_34/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_34
�
batch_normalization/beta_34
VariableV2*.
_class$
" loc:@batch_normalization/beta_34*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_34/AssignAssignbatch_normalization/beta_34-batch_normalization/beta_34/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_34*
validate_shape(
�
 batch_normalization/beta_34/readIdentitybatch_normalization/beta_34*
T0*.
_class$
" loc:@batch_normalization/beta_34
�
Dbatch_normalization/moving_mean_34/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_34*
dtype0
�
:batch_normalization/moving_mean_34/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_34*
dtype0
�
4batch_normalization/moving_mean_34/Initializer/zerosFillDbatch_normalization/moving_mean_34/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_34/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_34
�
"batch_normalization/moving_mean_34
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_34*
dtype0*
	container 
�
)batch_normalization/moving_mean_34/AssignAssign"batch_normalization/moving_mean_344batch_normalization/moving_mean_34/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_34*
validate_shape(
�
'batch_normalization/moving_mean_34/readIdentity"batch_normalization/moving_mean_34*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_34
�
Gbatch_normalization/moving_variance_34/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_34*
dtype0
�
=batch_normalization/moving_variance_34/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_34*
dtype0
�
7batch_normalization/moving_variance_34/Initializer/onesFillGbatch_normalization/moving_variance_34/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_34/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_34
�
&batch_normalization/moving_variance_34
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_34*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_34/AssignAssign&batch_normalization/moving_variance_347batch_normalization/moving_variance_34/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_34*
validate_shape(
�
+batch_normalization/moving_variance_34/readIdentity&batch_normalization/moving_variance_34*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_34
�
%batch_normalization/FusedBatchNorm_34FusedBatchNormres4d_branch2a!batch_normalization/gamma_34/read batch_normalization/beta_34/read'batch_normalization/moving_mean_34/read+batch_normalization/moving_variance_34/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_34Const*
valueB
 * �:*
dtype0
F
bn4d_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4d_branch2aIdentitybn4d_branch2a*
T0
6
res4d_branch2a_reluReluscale4d_branch2a*
T0
K
res4d_branch2b/kernelConst*
valueB@�*
dtype0
�
res4d_branch2bConv2Dres4d_branch2a_relures4d_branch2b/kernel*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_35/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_35*
valueB:�*
dtype0
�
3batch_normalization/gamma_35/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_35*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_35/Initializer/onesFill=batch_normalization/gamma_35/Initializer/ones/shape_as_tensor3batch_normalization/gamma_35/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_35*

index_type0
�
batch_normalization/gamma_35
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_35*
dtype0
�
#batch_normalization/gamma_35/AssignAssignbatch_normalization/gamma_35-batch_normalization/gamma_35/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_35*
validate_shape(
�
!batch_normalization/gamma_35/readIdentitybatch_normalization/gamma_35*
T0*/
_class%
#!loc:@batch_normalization/gamma_35
�
=batch_normalization/beta_35/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_35*
valueB:�*
dtype0
�
3batch_normalization/beta_35/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_35*
valueB
 *    *
dtype0
�
-batch_normalization/beta_35/Initializer/zerosFill=batch_normalization/beta_35/Initializer/zeros/shape_as_tensor3batch_normalization/beta_35/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_35*

index_type0
�
batch_normalization/beta_35
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_35
�
"batch_normalization/beta_35/AssignAssignbatch_normalization/beta_35-batch_normalization/beta_35/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_35*
validate_shape(*
use_locking(
�
 batch_normalization/beta_35/readIdentitybatch_normalization/beta_35*
T0*.
_class$
" loc:@batch_normalization/beta_35
�
Dbatch_normalization/moving_mean_35/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_35*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_35/Initializer/zeros/ConstConst*
dtype0*5
_class+
)'loc:@batch_normalization/moving_mean_35*
valueB
 *    
�
4batch_normalization/moving_mean_35/Initializer/zerosFillDbatch_normalization/moving_mean_35/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_35/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_35*

index_type0
�
"batch_normalization/moving_mean_35
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_35*
dtype0*
	container 
�
)batch_normalization/moving_mean_35/AssignAssign"batch_normalization/moving_mean_354batch_normalization/moving_mean_35/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_35*
validate_shape(
�
'batch_normalization/moving_mean_35/readIdentity"batch_normalization/moving_mean_35*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_35
�
Gbatch_normalization/moving_variance_35/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_35*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_35/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_35*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_35/Initializer/onesFillGbatch_normalization/moving_variance_35/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_35/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_35*

index_type0
�
&batch_normalization/moving_variance_35
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_35*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_35/AssignAssign&batch_normalization/moving_variance_357batch_normalization/moving_variance_35/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_35*
validate_shape(
�
+batch_normalization/moving_variance_35/readIdentity&batch_normalization/moving_variance_35*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_35
�
%batch_normalization/FusedBatchNorm_35FusedBatchNormres4d_branch2b!batch_normalization/gamma_35/read batch_normalization/beta_35/read'batch_normalization/moving_mean_35/read+batch_normalization/moving_variance_35/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_35Const*
valueB
 * �:*
dtype0
F
bn4d_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4d_branch2bIdentitybn4d_branch2b*
T0
6
res4d_branch2b_reluReluscale4d_branch2b*
T0
K
res4d_branch2c/kernelConst*
valueB@�*
dtype0
�
res4d_branch2cConv2Dres4d_branch2b_relures4d_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_36/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_36*
dtype0
�
3batch_normalization/gamma_36/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_36*
dtype0
�
-batch_normalization/gamma_36/Initializer/onesFill=batch_normalization/gamma_36/Initializer/ones/shape_as_tensor3batch_normalization/gamma_36/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_36
�
batch_normalization/gamma_36
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_36*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_36/AssignAssignbatch_normalization/gamma_36-batch_normalization/gamma_36/Initializer/ones*
validate_shape(*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_36
�
!batch_normalization/gamma_36/readIdentitybatch_normalization/gamma_36*
T0*/
_class%
#!loc:@batch_normalization/gamma_36
�
=batch_normalization/beta_36/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_36*
dtype0
�
3batch_normalization/beta_36/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_36*
dtype0
�
-batch_normalization/beta_36/Initializer/zerosFill=batch_normalization/beta_36/Initializer/zeros/shape_as_tensor3batch_normalization/beta_36/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_36
�
batch_normalization/beta_36
VariableV2*.
_class$
" loc:@batch_normalization/beta_36*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_36/AssignAssignbatch_normalization/beta_36-batch_normalization/beta_36/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_36
�
 batch_normalization/beta_36/readIdentitybatch_normalization/beta_36*
T0*.
_class$
" loc:@batch_normalization/beta_36
�
Dbatch_normalization/moving_mean_36/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_36*
dtype0
�
:batch_normalization/moving_mean_36/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_36*
dtype0
�
4batch_normalization/moving_mean_36/Initializer/zerosFillDbatch_normalization/moving_mean_36/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_36/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_36
�
"batch_normalization/moving_mean_36
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_36*
dtype0*
	container 
�
)batch_normalization/moving_mean_36/AssignAssign"batch_normalization/moving_mean_364batch_normalization/moving_mean_36/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_36
�
'batch_normalization/moving_mean_36/readIdentity"batch_normalization/moving_mean_36*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_36
�
Gbatch_normalization/moving_variance_36/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_36*
dtype0
�
=batch_normalization/moving_variance_36/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_36*
dtype0
�
7batch_normalization/moving_variance_36/Initializer/onesFillGbatch_normalization/moving_variance_36/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_36/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_36
�
&batch_normalization/moving_variance_36
VariableV2*
dtype0*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_36
�
-batch_normalization/moving_variance_36/AssignAssign&batch_normalization/moving_variance_367batch_normalization/moving_variance_36/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_36*
validate_shape(
�
+batch_normalization/moving_variance_36/readIdentity&batch_normalization/moving_variance_36*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_36
�
%batch_normalization/FusedBatchNorm_36FusedBatchNormres4d_branch2c!batch_normalization/gamma_36/read batch_normalization/beta_36/read'batch_normalization/moving_mean_36/read+batch_normalization/moving_variance_36/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_36Const*
valueB
 * �:*
dtype0
F
bn4d_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4d_branch2cIdentitybn4d_branch2c*
T0
3
res4dAdd
res4c_reluscale4d_branch2c*
T0
"

res4d_reluRelures4d*
T0
K
res4e_branch2a/kernelConst*
dtype0*
valueB@�
�
res4e_branch2aConv2D
res4d_relures4e_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_37/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_37*
valueB:�*
dtype0
�
3batch_normalization/gamma_37/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_37*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_37/Initializer/onesFill=batch_normalization/gamma_37/Initializer/ones/shape_as_tensor3batch_normalization/gamma_37/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_37*

index_type0
�
batch_normalization/gamma_37
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_37*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_37/AssignAssignbatch_normalization/gamma_37-batch_normalization/gamma_37/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_37*
validate_shape(
�
!batch_normalization/gamma_37/readIdentitybatch_normalization/gamma_37*
T0*/
_class%
#!loc:@batch_normalization/gamma_37
�
=batch_normalization/beta_37/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_37*
valueB:�*
dtype0
�
3batch_normalization/beta_37/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_37*
valueB
 *    *
dtype0
�
-batch_normalization/beta_37/Initializer/zerosFill=batch_normalization/beta_37/Initializer/zeros/shape_as_tensor3batch_normalization/beta_37/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_37*

index_type0
�
batch_normalization/beta_37
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_37*
dtype0*
	container 
�
"batch_normalization/beta_37/AssignAssignbatch_normalization/beta_37-batch_normalization/beta_37/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_37*
validate_shape(*
use_locking(
�
 batch_normalization/beta_37/readIdentitybatch_normalization/beta_37*.
_class$
" loc:@batch_normalization/beta_37*
T0
�
Dbatch_normalization/moving_mean_37/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_37*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_37/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_37*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_37/Initializer/zerosFillDbatch_normalization/moving_mean_37/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_37/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_37*

index_type0
�
"batch_normalization/moving_mean_37
VariableV2*5
_class+
)'loc:@batch_normalization/moving_mean_37*
dtype0*
	container *
shape:�*
shared_name 
�
)batch_normalization/moving_mean_37/AssignAssign"batch_normalization/moving_mean_374batch_normalization/moving_mean_37/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_37*
validate_shape(
�
'batch_normalization/moving_mean_37/readIdentity"batch_normalization/moving_mean_37*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_37
�
Gbatch_normalization/moving_variance_37/Initializer/ones/shape_as_tensorConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_37*
valueB:�
�
=batch_normalization/moving_variance_37/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_37*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_37/Initializer/onesFillGbatch_normalization/moving_variance_37/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_37/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_37*

index_type0
�
&batch_normalization/moving_variance_37
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_37*
dtype0*
	container 
�
-batch_normalization/moving_variance_37/AssignAssign&batch_normalization/moving_variance_377batch_normalization/moving_variance_37/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_37*
validate_shape(
�
+batch_normalization/moving_variance_37/readIdentity&batch_normalization/moving_variance_37*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_37
�
%batch_normalization/FusedBatchNorm_37FusedBatchNormres4e_branch2a!batch_normalization/gamma_37/read batch_normalization/beta_37/read'batch_normalization/moving_mean_37/read+batch_normalization/moving_variance_37/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_37Const*
valueB
 * �:*
dtype0
F
bn4e_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4e_branch2aIdentitybn4e_branch2a*
T0
6
res4e_branch2a_reluReluscale4e_branch2a*
T0
K
res4e_branch2b/kernelConst*
valueB@�*
dtype0
�
res4e_branch2bConv2Dres4e_branch2a_relures4e_branch2b/kernel*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
�
=batch_normalization/gamma_38/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_38*
dtype0
�
3batch_normalization/gamma_38/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_38*
dtype0
�
-batch_normalization/gamma_38/Initializer/onesFill=batch_normalization/gamma_38/Initializer/ones/shape_as_tensor3batch_normalization/gamma_38/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_38
�
batch_normalization/gamma_38
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_38*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_38/AssignAssignbatch_normalization/gamma_38-batch_normalization/gamma_38/Initializer/ones*
validate_shape(*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_38
�
!batch_normalization/gamma_38/readIdentitybatch_normalization/gamma_38*
T0*/
_class%
#!loc:@batch_normalization/gamma_38
�
=batch_normalization/beta_38/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_38*
dtype0
�
3batch_normalization/beta_38/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_38*
dtype0
�
-batch_normalization/beta_38/Initializer/zerosFill=batch_normalization/beta_38/Initializer/zeros/shape_as_tensor3batch_normalization/beta_38/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_38
�
batch_normalization/beta_38
VariableV2*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_38*
dtype0
�
"batch_normalization/beta_38/AssignAssignbatch_normalization/beta_38-batch_normalization/beta_38/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_38*
validate_shape(
�
 batch_normalization/beta_38/readIdentitybatch_normalization/beta_38*
T0*.
_class$
" loc:@batch_normalization/beta_38
�
Dbatch_normalization/moving_mean_38/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_38*
dtype0
�
:batch_normalization/moving_mean_38/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_38
�
4batch_normalization/moving_mean_38/Initializer/zerosFillDbatch_normalization/moving_mean_38/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_38/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_38
�
"batch_normalization/moving_mean_38
VariableV2*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_38*
dtype0
�
)batch_normalization/moving_mean_38/AssignAssign"batch_normalization/moving_mean_384batch_normalization/moving_mean_38/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_38*
validate_shape(
�
'batch_normalization/moving_mean_38/readIdentity"batch_normalization/moving_mean_38*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_38
�
Gbatch_normalization/moving_variance_38/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_38*
dtype0
�
=batch_normalization/moving_variance_38/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_38*
dtype0
�
7batch_normalization/moving_variance_38/Initializer/onesFillGbatch_normalization/moving_variance_38/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_38/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_38
�
&batch_normalization/moving_variance_38
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_38*
dtype0*
	container 
�
-batch_normalization/moving_variance_38/AssignAssign&batch_normalization/moving_variance_387batch_normalization/moving_variance_38/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_38*
validate_shape(
�
+batch_normalization/moving_variance_38/readIdentity&batch_normalization/moving_variance_38*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_38
�
%batch_normalization/FusedBatchNorm_38FusedBatchNormres4e_branch2b!batch_normalization/gamma_38/read batch_normalization/beta_38/read'batch_normalization/moving_mean_38/read+batch_normalization/moving_variance_38/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_38Const*
valueB
 * �:*
dtype0
F
bn4e_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4e_branch2bIdentitybn4e_branch2b*
T0
6
res4e_branch2b_reluReluscale4e_branch2b*
T0
K
res4e_branch2c/kernelConst*
valueB@�*
dtype0
�
res4e_branch2cConv2Dres4e_branch2b_relures4e_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_39/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_39*
valueB:�*
dtype0
�
3batch_normalization/gamma_39/Initializer/ones/ConstConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_39*
valueB
 *  �?
�
-batch_normalization/gamma_39/Initializer/onesFill=batch_normalization/gamma_39/Initializer/ones/shape_as_tensor3batch_normalization/gamma_39/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_39*

index_type0
�
batch_normalization/gamma_39
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_39*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_39/AssignAssignbatch_normalization/gamma_39-batch_normalization/gamma_39/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_39*
validate_shape(
�
!batch_normalization/gamma_39/readIdentitybatch_normalization/gamma_39*
T0*/
_class%
#!loc:@batch_normalization/gamma_39
�
=batch_normalization/beta_39/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_39*
valueB:�*
dtype0
�
3batch_normalization/beta_39/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_39*
valueB
 *    *
dtype0
�
-batch_normalization/beta_39/Initializer/zerosFill=batch_normalization/beta_39/Initializer/zeros/shape_as_tensor3batch_normalization/beta_39/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_39*

index_type0
�
batch_normalization/beta_39
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_39*
dtype0*
	container 
�
"batch_normalization/beta_39/AssignAssignbatch_normalization/beta_39-batch_normalization/beta_39/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_39*
validate_shape(
�
 batch_normalization/beta_39/readIdentitybatch_normalization/beta_39*
T0*.
_class$
" loc:@batch_normalization/beta_39
�
Dbatch_normalization/moving_mean_39/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_39*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_39/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_39*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_39/Initializer/zerosFillDbatch_normalization/moving_mean_39/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_39/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_39*

index_type0
�
"batch_normalization/moving_mean_39
VariableV2*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_39*
dtype0
�
)batch_normalization/moving_mean_39/AssignAssign"batch_normalization/moving_mean_394batch_normalization/moving_mean_39/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_39*
validate_shape(
�
'batch_normalization/moving_mean_39/readIdentity"batch_normalization/moving_mean_39*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_39
�
Gbatch_normalization/moving_variance_39/Initializer/ones/shape_as_tensorConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_39*
valueB:�
�
=batch_normalization/moving_variance_39/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_39*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_39/Initializer/onesFillGbatch_normalization/moving_variance_39/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_39/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_39*

index_type0
�
&batch_normalization/moving_variance_39
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_39*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_39/AssignAssign&batch_normalization/moving_variance_397batch_normalization/moving_variance_39/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_39*
validate_shape(
�
+batch_normalization/moving_variance_39/readIdentity&batch_normalization/moving_variance_39*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_39
�
%batch_normalization/FusedBatchNorm_39FusedBatchNormres4e_branch2c!batch_normalization/gamma_39/read batch_normalization/beta_39/read'batch_normalization/moving_mean_39/read+batch_normalization/moving_variance_39/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_39Const*
valueB
 * �:*
dtype0
F
bn4e_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4e_branch2cIdentitybn4e_branch2c*
T0
3
res4eAdd
res4d_reluscale4e_branch2c*
T0
"

res4e_reluRelures4e*
T0
K
res4f_branch2a/kernelConst*
valueB@�*
dtype0
�
res4f_branch2aConv2D
res4e_relures4f_branch2a/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_40/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_40*
dtype0
�
3batch_normalization/gamma_40/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_40*
dtype0
�
-batch_normalization/gamma_40/Initializer/onesFill=batch_normalization/gamma_40/Initializer/ones/shape_as_tensor3batch_normalization/gamma_40/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_40
�
batch_normalization/gamma_40
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_40*
dtype0
�
#batch_normalization/gamma_40/AssignAssignbatch_normalization/gamma_40-batch_normalization/gamma_40/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_40*
validate_shape(
�
!batch_normalization/gamma_40/readIdentitybatch_normalization/gamma_40*
T0*/
_class%
#!loc:@batch_normalization/gamma_40
�
=batch_normalization/beta_40/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_40*
dtype0
�
3batch_normalization/beta_40/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_40*
dtype0
�
-batch_normalization/beta_40/Initializer/zerosFill=batch_normalization/beta_40/Initializer/zeros/shape_as_tensor3batch_normalization/beta_40/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_40
�
batch_normalization/beta_40
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_40*
dtype0*
	container 
�
"batch_normalization/beta_40/AssignAssignbatch_normalization/beta_40-batch_normalization/beta_40/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_40*
validate_shape(*
use_locking(
�
 batch_normalization/beta_40/readIdentitybatch_normalization/beta_40*.
_class$
" loc:@batch_normalization/beta_40*
T0
�
Dbatch_normalization/moving_mean_40/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_40*
dtype0
�
:batch_normalization/moving_mean_40/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_40*
dtype0
�
4batch_normalization/moving_mean_40/Initializer/zerosFillDbatch_normalization/moving_mean_40/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_40/Initializer/zeros/Const*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_40*
T0
�
"batch_normalization/moving_mean_40
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_40*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_40/AssignAssign"batch_normalization/moving_mean_404batch_normalization/moving_mean_40/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_40
�
'batch_normalization/moving_mean_40/readIdentity"batch_normalization/moving_mean_40*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_40
�
Gbatch_normalization/moving_variance_40/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_40*
dtype0
�
=batch_normalization/moving_variance_40/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_40*
dtype0
�
7batch_normalization/moving_variance_40/Initializer/onesFillGbatch_normalization/moving_variance_40/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_40/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_40
�
&batch_normalization/moving_variance_40
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_40*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_40/AssignAssign&batch_normalization/moving_variance_407batch_normalization/moving_variance_40/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_40*
validate_shape(
�
+batch_normalization/moving_variance_40/readIdentity&batch_normalization/moving_variance_40*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_40
�
%batch_normalization/FusedBatchNorm_40FusedBatchNormres4f_branch2a!batch_normalization/gamma_40/read batch_normalization/beta_40/read'batch_normalization/moving_mean_40/read+batch_normalization/moving_variance_40/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_40Const*
valueB
 * �:*
dtype0
F
bn4f_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4f_branch2aIdentitybn4f_branch2a*
T0
6
res4f_branch2a_reluReluscale4f_branch2a*
T0
K
res4f_branch2b/kernelConst*
valueB@�*
dtype0
�
res4f_branch2bConv2Dres4f_branch2a_relures4f_branch2b/kernel*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0
�
=batch_normalization/gamma_41/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_41*
valueB:�*
dtype0
�
3batch_normalization/gamma_41/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_41*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_41/Initializer/onesFill=batch_normalization/gamma_41/Initializer/ones/shape_as_tensor3batch_normalization/gamma_41/Initializer/ones/Const*/
_class%
#!loc:@batch_normalization/gamma_41*

index_type0*
T0
�
batch_normalization/gamma_41
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_41*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_41/AssignAssignbatch_normalization/gamma_41-batch_normalization/gamma_41/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_41*
validate_shape(
�
!batch_normalization/gamma_41/readIdentitybatch_normalization/gamma_41*
T0*/
_class%
#!loc:@batch_normalization/gamma_41
�
=batch_normalization/beta_41/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_41*
valueB:�*
dtype0
�
3batch_normalization/beta_41/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_41*
valueB
 *    *
dtype0
�
-batch_normalization/beta_41/Initializer/zerosFill=batch_normalization/beta_41/Initializer/zeros/shape_as_tensor3batch_normalization/beta_41/Initializer/zeros/Const*.
_class$
" loc:@batch_normalization/beta_41*

index_type0*
T0
�
batch_normalization/beta_41
VariableV2*.
_class$
" loc:@batch_normalization/beta_41*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_41/AssignAssignbatch_normalization/beta_41-batch_normalization/beta_41/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_41*
validate_shape(*
use_locking(
�
 batch_normalization/beta_41/readIdentitybatch_normalization/beta_41*
T0*.
_class$
" loc:@batch_normalization/beta_41
�
Dbatch_normalization/moving_mean_41/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_41*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_41/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_41*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_41/Initializer/zerosFillDbatch_normalization/moving_mean_41/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_41/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_41*

index_type0
�
"batch_normalization/moving_mean_41
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_41*
dtype0*
	container 
�
)batch_normalization/moving_mean_41/AssignAssign"batch_normalization/moving_mean_414batch_normalization/moving_mean_41/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_41
�
'batch_normalization/moving_mean_41/readIdentity"batch_normalization/moving_mean_41*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_41
�
Gbatch_normalization/moving_variance_41/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_41*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_41/Initializer/ones/ConstConst*
dtype0*9
_class/
-+loc:@batch_normalization/moving_variance_41*
valueB
 *  �?
�
7batch_normalization/moving_variance_41/Initializer/onesFillGbatch_normalization/moving_variance_41/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_41/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_41*

index_type0
�
&batch_normalization/moving_variance_41
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_41*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_41/AssignAssign&batch_normalization/moving_variance_417batch_normalization/moving_variance_41/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_41*
validate_shape(
�
+batch_normalization/moving_variance_41/readIdentity&batch_normalization/moving_variance_41*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_41
�
%batch_normalization/FusedBatchNorm_41FusedBatchNormres4f_branch2b!batch_normalization/gamma_41/read batch_normalization/beta_41/read'batch_normalization/moving_mean_41/read+batch_normalization/moving_variance_41/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_41Const*
valueB
 * �:*
dtype0
F
bn4f_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4f_branch2bIdentitybn4f_branch2b*
T0
6
res4f_branch2b_reluReluscale4f_branch2b*
T0
K
res4f_branch2c/kernelConst*
valueB@�*
dtype0
�
res4f_branch2cConv2Dres4f_branch2b_relures4f_branch2c/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_42/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_42*
dtype0
�
3batch_normalization/gamma_42/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_42*
dtype0
�
-batch_normalization/gamma_42/Initializer/onesFill=batch_normalization/gamma_42/Initializer/ones/shape_as_tensor3batch_normalization/gamma_42/Initializer/ones/Const*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_42*
T0
�
batch_normalization/gamma_42
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_42*
dtype0
�
#batch_normalization/gamma_42/AssignAssignbatch_normalization/gamma_42-batch_normalization/gamma_42/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_42*
validate_shape(
�
!batch_normalization/gamma_42/readIdentitybatch_normalization/gamma_42*
T0*/
_class%
#!loc:@batch_normalization/gamma_42
�
=batch_normalization/beta_42/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_42*
dtype0
�
3batch_normalization/beta_42/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_42*
dtype0
�
-batch_normalization/beta_42/Initializer/zerosFill=batch_normalization/beta_42/Initializer/zeros/shape_as_tensor3batch_normalization/beta_42/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_42
�
batch_normalization/beta_42
VariableV2*
dtype0*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_42
�
"batch_normalization/beta_42/AssignAssignbatch_normalization/beta_42-batch_normalization/beta_42/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_42*
validate_shape(
�
 batch_normalization/beta_42/readIdentitybatch_normalization/beta_42*
T0*.
_class$
" loc:@batch_normalization/beta_42
�
Dbatch_normalization/moving_mean_42/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_42*
dtype0
�
:batch_normalization/moving_mean_42/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_42*
dtype0
�
4batch_normalization/moving_mean_42/Initializer/zerosFillDbatch_normalization/moving_mean_42/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_42/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_42
�
"batch_normalization/moving_mean_42
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_42*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_42/AssignAssign"batch_normalization/moving_mean_424batch_normalization/moving_mean_42/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_42*
validate_shape(
�
'batch_normalization/moving_mean_42/readIdentity"batch_normalization/moving_mean_42*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_42
�
Gbatch_normalization/moving_variance_42/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_42*
dtype0
�
=batch_normalization/moving_variance_42/Initializer/ones/ConstConst*
dtype0*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_42
�
7batch_normalization/moving_variance_42/Initializer/onesFillGbatch_normalization/moving_variance_42/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_42/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_42
�
&batch_normalization/moving_variance_42
VariableV2*
dtype0*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_42
�
-batch_normalization/moving_variance_42/AssignAssign&batch_normalization/moving_variance_427batch_normalization/moving_variance_42/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_42*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_42/readIdentity&batch_normalization/moving_variance_42*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_42
�
%batch_normalization/FusedBatchNorm_42FusedBatchNormres4f_branch2c!batch_normalization/gamma_42/read batch_normalization/beta_42/read'batch_normalization/moving_mean_42/read+batch_normalization/moving_variance_42/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_42Const*
valueB
 * �:*
dtype0
F
bn4f_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale4f_branch2cIdentitybn4f_branch2c*
T0
3
res4fAdd
res4e_reluscale4f_branch2c*
T0
"

res4f_reluRelures4f*
T0
J
res5a_branch1/kernelConst*
valueB@�*
dtype0
�
res5a_branch1Conv2D
res4f_relures5a_branch1/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_43/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_43*
valueB:�*
dtype0
�
3batch_normalization/gamma_43/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_43*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_43/Initializer/onesFill=batch_normalization/gamma_43/Initializer/ones/shape_as_tensor3batch_normalization/gamma_43/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_43*

index_type0
�
batch_normalization/gamma_43
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_43*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_43/AssignAssignbatch_normalization/gamma_43-batch_normalization/gamma_43/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_43*
validate_shape(
�
!batch_normalization/gamma_43/readIdentitybatch_normalization/gamma_43*
T0*/
_class%
#!loc:@batch_normalization/gamma_43
�
=batch_normalization/beta_43/Initializer/zeros/shape_as_tensorConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_43*
valueB:�
�
3batch_normalization/beta_43/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_43*
valueB
 *    *
dtype0
�
-batch_normalization/beta_43/Initializer/zerosFill=batch_normalization/beta_43/Initializer/zeros/shape_as_tensor3batch_normalization/beta_43/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_43*

index_type0
�
batch_normalization/beta_43
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_43*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_43/AssignAssignbatch_normalization/beta_43-batch_normalization/beta_43/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_43
�
 batch_normalization/beta_43/readIdentitybatch_normalization/beta_43*
T0*.
_class$
" loc:@batch_normalization/beta_43
�
Dbatch_normalization/moving_mean_43/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_43*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_43/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_43*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_43/Initializer/zerosFillDbatch_normalization/moving_mean_43/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_43/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_43*

index_type0
�
"batch_normalization/moving_mean_43
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_43*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_43/AssignAssign"batch_normalization/moving_mean_434batch_normalization/moving_mean_43/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_43*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_43/readIdentity"batch_normalization/moving_mean_43*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_43
�
Gbatch_normalization/moving_variance_43/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_43*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_43/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_43*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_43/Initializer/onesFillGbatch_normalization/moving_variance_43/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_43/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_43*

index_type0
�
&batch_normalization/moving_variance_43
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_43*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_43/AssignAssign&batch_normalization/moving_variance_437batch_normalization/moving_variance_43/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_43*
validate_shape(
�
+batch_normalization/moving_variance_43/readIdentity&batch_normalization/moving_variance_43*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_43
�
%batch_normalization/FusedBatchNorm_43FusedBatchNormres5a_branch1!batch_normalization/gamma_43/read batch_normalization/beta_43/read'batch_normalization/moving_mean_43/read+batch_normalization/moving_variance_43/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_43Const*
valueB
 * �:*
dtype0
E
bn5a_branch1Identity"batch_normalization/FusedBatchNorm*
T0
2
scale5a_branch1Identitybn5a_branch1*
T0
K
res5a_branch2a/kernelConst*
valueB@�*
dtype0
�
res5a_branch2aConv2D
res4f_relures5a_branch2a/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_44/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_44*
dtype0
�
3batch_normalization/gamma_44/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_44*
dtype0
�
-batch_normalization/gamma_44/Initializer/onesFill=batch_normalization/gamma_44/Initializer/ones/shape_as_tensor3batch_normalization/gamma_44/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_44
�
batch_normalization/gamma_44
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_44*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_44/AssignAssignbatch_normalization/gamma_44-batch_normalization/gamma_44/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_44*
validate_shape(
�
!batch_normalization/gamma_44/readIdentitybatch_normalization/gamma_44*
T0*/
_class%
#!loc:@batch_normalization/gamma_44
�
=batch_normalization/beta_44/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:�*.
_class$
" loc:@batch_normalization/beta_44
�
3batch_normalization/beta_44/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_44*
dtype0
�
-batch_normalization/beta_44/Initializer/zerosFill=batch_normalization/beta_44/Initializer/zeros/shape_as_tensor3batch_normalization/beta_44/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_44
�
batch_normalization/beta_44
VariableV2*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_44*
dtype0
�
"batch_normalization/beta_44/AssignAssignbatch_normalization/beta_44-batch_normalization/beta_44/Initializer/zeros*
T0*.
_class$
" loc:@batch_normalization/beta_44*
validate_shape(*
use_locking(
�
 batch_normalization/beta_44/readIdentitybatch_normalization/beta_44*
T0*.
_class$
" loc:@batch_normalization/beta_44
�
Dbatch_normalization/moving_mean_44/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_44*
dtype0
�
:batch_normalization/moving_mean_44/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_44*
dtype0
�
4batch_normalization/moving_mean_44/Initializer/zerosFillDbatch_normalization/moving_mean_44/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_44/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_44
�
"batch_normalization/moving_mean_44
VariableV2*
dtype0*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_44
�
)batch_normalization/moving_mean_44/AssignAssign"batch_normalization/moving_mean_444batch_normalization/moving_mean_44/Initializer/zeros*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_44*
validate_shape(*
use_locking(
�
'batch_normalization/moving_mean_44/readIdentity"batch_normalization/moving_mean_44*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_44
�
Gbatch_normalization/moving_variance_44/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_44*
dtype0
�
=batch_normalization/moving_variance_44/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_44*
dtype0
�
7batch_normalization/moving_variance_44/Initializer/onesFillGbatch_normalization/moving_variance_44/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_44/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_44
�
&batch_normalization/moving_variance_44
VariableV2*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_44*
dtype0*
	container *
shape:�
�
-batch_normalization/moving_variance_44/AssignAssign&batch_normalization/moving_variance_447batch_normalization/moving_variance_44/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_44*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_44/readIdentity&batch_normalization/moving_variance_44*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_44
�
%batch_normalization/FusedBatchNorm_44FusedBatchNormres5a_branch2a!batch_normalization/gamma_44/read batch_normalization/beta_44/read'batch_normalization/moving_mean_44/read+batch_normalization/moving_variance_44/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_44Const*
valueB
 * �:*
dtype0
F
bn5a_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5a_branch2aIdentitybn5a_branch2a*
T0
6
res5a_branch2a_reluReluscale5a_branch2a*
T0
K
res5a_branch2b/kernelConst*
valueB@�*
dtype0
�
res5a_branch2bConv2Dres5a_branch2a_relures5a_branch2b/kernel*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
=batch_normalization/gamma_45/Initializer/ones/shape_as_tensorConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_45*
valueB:�
�
3batch_normalization/gamma_45/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_45*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_45/Initializer/onesFill=batch_normalization/gamma_45/Initializer/ones/shape_as_tensor3batch_normalization/gamma_45/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_45*

index_type0
�
batch_normalization/gamma_45
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_45*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_45/AssignAssignbatch_normalization/gamma_45-batch_normalization/gamma_45/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_45*
validate_shape(
�
!batch_normalization/gamma_45/readIdentitybatch_normalization/gamma_45*
T0*/
_class%
#!loc:@batch_normalization/gamma_45
�
=batch_normalization/beta_45/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_45*
valueB:�*
dtype0
�
3batch_normalization/beta_45/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_45*
valueB
 *    *
dtype0
�
-batch_normalization/beta_45/Initializer/zerosFill=batch_normalization/beta_45/Initializer/zeros/shape_as_tensor3batch_normalization/beta_45/Initializer/zeros/Const*.
_class$
" loc:@batch_normalization/beta_45*

index_type0*
T0
�
batch_normalization/beta_45
VariableV2*.
_class$
" loc:@batch_normalization/beta_45*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_45/AssignAssignbatch_normalization/beta_45-batch_normalization/beta_45/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_45*
validate_shape(
�
 batch_normalization/beta_45/readIdentitybatch_normalization/beta_45*
T0*.
_class$
" loc:@batch_normalization/beta_45
�
Dbatch_normalization/moving_mean_45/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_45*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_45/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_45*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_45/Initializer/zerosFillDbatch_normalization/moving_mean_45/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_45/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_45*

index_type0
�
"batch_normalization/moving_mean_45
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_45*
dtype0*
	container 
�
)batch_normalization/moving_mean_45/AssignAssign"batch_normalization/moving_mean_454batch_normalization/moving_mean_45/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_45*
validate_shape(
�
'batch_normalization/moving_mean_45/readIdentity"batch_normalization/moving_mean_45*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_45
�
Gbatch_normalization/moving_variance_45/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_45*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_45/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_45*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_45/Initializer/onesFillGbatch_normalization/moving_variance_45/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_45/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_45*

index_type0
�
&batch_normalization/moving_variance_45
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_45*
dtype0*
	container 
�
-batch_normalization/moving_variance_45/AssignAssign&batch_normalization/moving_variance_457batch_normalization/moving_variance_45/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_45*
validate_shape(
�
+batch_normalization/moving_variance_45/readIdentity&batch_normalization/moving_variance_45*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_45
�
%batch_normalization/FusedBatchNorm_45FusedBatchNormres5a_branch2b!batch_normalization/gamma_45/read batch_normalization/beta_45/read'batch_normalization/moving_mean_45/read+batch_normalization/moving_variance_45/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_45Const*
valueB
 * �:*
dtype0
F
bn5a_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5a_branch2bIdentitybn5a_branch2b*
T0
6
res5a_branch2b_reluReluscale5a_branch2b*
T0
K
res5a_branch2c/kernelConst*
valueB@�*
dtype0
�
res5a_branch2cConv2Dres5a_branch2b_relures5a_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_46/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_46*
dtype0
�
3batch_normalization/gamma_46/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_46*
dtype0
�
-batch_normalization/gamma_46/Initializer/onesFill=batch_normalization/gamma_46/Initializer/ones/shape_as_tensor3batch_normalization/gamma_46/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_46
�
batch_normalization/gamma_46
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_46*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_46/AssignAssignbatch_normalization/gamma_46-batch_normalization/gamma_46/Initializer/ones*
validate_shape(*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_46
�
!batch_normalization/gamma_46/readIdentitybatch_normalization/gamma_46*
T0*/
_class%
#!loc:@batch_normalization/gamma_46
�
=batch_normalization/beta_46/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_46*
dtype0
�
3batch_normalization/beta_46/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_46*
dtype0
�
-batch_normalization/beta_46/Initializer/zerosFill=batch_normalization/beta_46/Initializer/zeros/shape_as_tensor3batch_normalization/beta_46/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_46
�
batch_normalization/beta_46
VariableV2*
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_46*
dtype0*
	container 
�
"batch_normalization/beta_46/AssignAssignbatch_normalization/beta_46-batch_normalization/beta_46/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_46*
validate_shape(
�
 batch_normalization/beta_46/readIdentitybatch_normalization/beta_46*.
_class$
" loc:@batch_normalization/beta_46*
T0
�
Dbatch_normalization/moving_mean_46/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_46*
dtype0
�
:batch_normalization/moving_mean_46/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_46*
dtype0
�
4batch_normalization/moving_mean_46/Initializer/zerosFillDbatch_normalization/moving_mean_46/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_46/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_46
�
"batch_normalization/moving_mean_46
VariableV2*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_46*
dtype0
�
)batch_normalization/moving_mean_46/AssignAssign"batch_normalization/moving_mean_464batch_normalization/moving_mean_46/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_46*
validate_shape(
�
'batch_normalization/moving_mean_46/readIdentity"batch_normalization/moving_mean_46*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_46
�
Gbatch_normalization/moving_variance_46/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_46*
dtype0
�
=batch_normalization/moving_variance_46/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_46*
dtype0
�
7batch_normalization/moving_variance_46/Initializer/onesFillGbatch_normalization/moving_variance_46/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_46/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_46
�
&batch_normalization/moving_variance_46
VariableV2*
dtype0*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_46
�
-batch_normalization/moving_variance_46/AssignAssign&batch_normalization/moving_variance_467batch_normalization/moving_variance_46/Initializer/ones*
validate_shape(*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_46
�
+batch_normalization/moving_variance_46/readIdentity&batch_normalization/moving_variance_46*9
_class/
-+loc:@batch_normalization/moving_variance_46*
T0
�
%batch_normalization/FusedBatchNorm_46FusedBatchNormres5a_branch2c!batch_normalization/gamma_46/read batch_normalization/beta_46/read'batch_normalization/moving_mean_46/read+batch_normalization/moving_variance_46/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_46Const*
valueB
 * �:*
dtype0
F
bn5a_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5a_branch2cIdentitybn5a_branch2c*
T0
8
res5aAddscale5a_branch1scale5a_branch2c*
T0
"

res5a_reluRelures5a*
T0
K
res5b_branch2a/kernelConst*
valueB@�*
dtype0
�
res5b_branch2aConv2D
res5a_relures5b_branch2a/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_47/Initializer/ones/shape_as_tensorConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_47*
valueB:�
�
3batch_normalization/gamma_47/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_47*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_47/Initializer/onesFill=batch_normalization/gamma_47/Initializer/ones/shape_as_tensor3batch_normalization/gamma_47/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_47*

index_type0
�
batch_normalization/gamma_47
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_47*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_47/AssignAssignbatch_normalization/gamma_47-batch_normalization/gamma_47/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_47*
validate_shape(
�
!batch_normalization/gamma_47/readIdentitybatch_normalization/gamma_47*
T0*/
_class%
#!loc:@batch_normalization/gamma_47
�
=batch_normalization/beta_47/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_47*
valueB:�*
dtype0
�
3batch_normalization/beta_47/Initializer/zeros/ConstConst*
dtype0*.
_class$
" loc:@batch_normalization/beta_47*
valueB
 *    
�
-batch_normalization/beta_47/Initializer/zerosFill=batch_normalization/beta_47/Initializer/zeros/shape_as_tensor3batch_normalization/beta_47/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_47*

index_type0
�
batch_normalization/beta_47
VariableV2*.
_class$
" loc:@batch_normalization/beta_47*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_47/AssignAssignbatch_normalization/beta_47-batch_normalization/beta_47/Initializer/zeros*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_47
�
 batch_normalization/beta_47/readIdentitybatch_normalization/beta_47*
T0*.
_class$
" loc:@batch_normalization/beta_47
�
Dbatch_normalization/moving_mean_47/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_47*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_47/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_47*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_47/Initializer/zerosFillDbatch_normalization/moving_mean_47/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_47/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_47*

index_type0
�
"batch_normalization/moving_mean_47
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_47*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_47/AssignAssign"batch_normalization/moving_mean_474batch_normalization/moving_mean_47/Initializer/zeros*
validate_shape(*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_47
�
'batch_normalization/moving_mean_47/readIdentity"batch_normalization/moving_mean_47*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_47
�
Gbatch_normalization/moving_variance_47/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_47*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_47/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_47*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_47/Initializer/onesFillGbatch_normalization/moving_variance_47/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_47/Initializer/ones/Const*9
_class/
-+loc:@batch_normalization/moving_variance_47*

index_type0*
T0
�
&batch_normalization/moving_variance_47
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_47*
dtype0*
	container 
�
-batch_normalization/moving_variance_47/AssignAssign&batch_normalization/moving_variance_477batch_normalization/moving_variance_47/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_47*
validate_shape(
�
+batch_normalization/moving_variance_47/readIdentity&batch_normalization/moving_variance_47*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_47
�
%batch_normalization/FusedBatchNorm_47FusedBatchNormres5b_branch2a!batch_normalization/gamma_47/read batch_normalization/beta_47/read'batch_normalization/moving_mean_47/read+batch_normalization/moving_variance_47/read*
T0*
data_formatNHWC*
is_training( *
epsilon%��'7
I
batch_normalization/Const_47Const*
dtype0*
valueB
 * �:
F
bn5b_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5b_branch2aIdentitybn5b_branch2a*
T0
6
res5b_branch2a_reluReluscale5b_branch2a*
T0
K
res5b_branch2b/kernelConst*
valueB@�*
dtype0
�
res5b_branch2bConv2Dres5b_branch2a_relures5b_branch2b/kernel*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

�
=batch_normalization/gamma_48/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_48
�
3batch_normalization/gamma_48/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_48*
dtype0
�
-batch_normalization/gamma_48/Initializer/onesFill=batch_normalization/gamma_48/Initializer/ones/shape_as_tensor3batch_normalization/gamma_48/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_48
�
batch_normalization/gamma_48
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_48*
dtype0*
	container 
�
#batch_normalization/gamma_48/AssignAssignbatch_normalization/gamma_48-batch_normalization/gamma_48/Initializer/ones*
validate_shape(*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_48
�
!batch_normalization/gamma_48/readIdentitybatch_normalization/gamma_48*
T0*/
_class%
#!loc:@batch_normalization/gamma_48
�
=batch_normalization/beta_48/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_48*
dtype0
�
3batch_normalization/beta_48/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_48
�
-batch_normalization/beta_48/Initializer/zerosFill=batch_normalization/beta_48/Initializer/zeros/shape_as_tensor3batch_normalization/beta_48/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_48
�
batch_normalization/beta_48
VariableV2*.
_class$
" loc:@batch_normalization/beta_48*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_48/AssignAssignbatch_normalization/beta_48-batch_normalization/beta_48/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_48*
validate_shape(
�
 batch_normalization/beta_48/readIdentitybatch_normalization/beta_48*
T0*.
_class$
" loc:@batch_normalization/beta_48
�
Dbatch_normalization/moving_mean_48/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_48*
dtype0
�
:batch_normalization/moving_mean_48/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_48*
dtype0
�
4batch_normalization/moving_mean_48/Initializer/zerosFillDbatch_normalization/moving_mean_48/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_48/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_48
�
"batch_normalization/moving_mean_48
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_48*
dtype0*
	container 
�
)batch_normalization/moving_mean_48/AssignAssign"batch_normalization/moving_mean_484batch_normalization/moving_mean_48/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_48*
validate_shape(
�
'batch_normalization/moving_mean_48/readIdentity"batch_normalization/moving_mean_48*5
_class+
)'loc:@batch_normalization/moving_mean_48*
T0
�
Gbatch_normalization/moving_variance_48/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_48*
dtype0
�
=batch_normalization/moving_variance_48/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_48*
dtype0
�
7batch_normalization/moving_variance_48/Initializer/onesFillGbatch_normalization/moving_variance_48/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_48/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_48
�
&batch_normalization/moving_variance_48
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_48*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_48/AssignAssign&batch_normalization/moving_variance_487batch_normalization/moving_variance_48/Initializer/ones*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_48*
validate_shape(*
use_locking(
�
+batch_normalization/moving_variance_48/readIdentity&batch_normalization/moving_variance_48*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_48
�
%batch_normalization/FusedBatchNorm_48FusedBatchNormres5b_branch2b!batch_normalization/gamma_48/read batch_normalization/beta_48/read'batch_normalization/moving_mean_48/read+batch_normalization/moving_variance_48/read*
is_training( *
epsilon%��'7*
T0*
data_formatNHWC
I
batch_normalization/Const_48Const*
valueB
 * �:*
dtype0
F
bn5b_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5b_branch2bIdentitybn5b_branch2b*
T0
6
res5b_branch2b_reluReluscale5b_branch2b*
T0
K
res5b_branch2c/kernelConst*
valueB@�*
dtype0
�
res5b_branch2cConv2Dres5b_branch2b_relures5b_branch2c/kernel*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
	dilations
*
T0
�
=batch_normalization/gamma_49/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_49*
valueB:�*
dtype0
�
3batch_normalization/gamma_49/Initializer/ones/ConstConst*
dtype0*/
_class%
#!loc:@batch_normalization/gamma_49*
valueB
 *  �?
�
-batch_normalization/gamma_49/Initializer/onesFill=batch_normalization/gamma_49/Initializer/ones/shape_as_tensor3batch_normalization/gamma_49/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_49*

index_type0
�
batch_normalization/gamma_49
VariableV2*
	container *
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_49*
dtype0
�
#batch_normalization/gamma_49/AssignAssignbatch_normalization/gamma_49-batch_normalization/gamma_49/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_49*
validate_shape(
�
!batch_normalization/gamma_49/readIdentitybatch_normalization/gamma_49*/
_class%
#!loc:@batch_normalization/gamma_49*
T0
�
=batch_normalization/beta_49/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_49*
valueB:�*
dtype0
�
3batch_normalization/beta_49/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_49*
valueB
 *    *
dtype0
�
-batch_normalization/beta_49/Initializer/zerosFill=batch_normalization/beta_49/Initializer/zeros/shape_as_tensor3batch_normalization/beta_49/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_49*

index_type0
�
batch_normalization/beta_49
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_49*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_49/AssignAssignbatch_normalization/beta_49-batch_normalization/beta_49/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_49*
validate_shape(
�
 batch_normalization/beta_49/readIdentitybatch_normalization/beta_49*
T0*.
_class$
" loc:@batch_normalization/beta_49
�
Dbatch_normalization/moving_mean_49/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_49*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_49/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_49*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_49/Initializer/zerosFillDbatch_normalization/moving_mean_49/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_49/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_49*

index_type0
�
"batch_normalization/moving_mean_49
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_49*
dtype0*
	container 
�
)batch_normalization/moving_mean_49/AssignAssign"batch_normalization/moving_mean_494batch_normalization/moving_mean_49/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_49*
validate_shape(
�
'batch_normalization/moving_mean_49/readIdentity"batch_normalization/moving_mean_49*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_49
�
Gbatch_normalization/moving_variance_49/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_49*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_49/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_49*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_49/Initializer/onesFillGbatch_normalization/moving_variance_49/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_49/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_49*

index_type0
�
&batch_normalization/moving_variance_49
VariableV2*
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_49*
dtype0*
	container 
�
-batch_normalization/moving_variance_49/AssignAssign&batch_normalization/moving_variance_497batch_normalization/moving_variance_49/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_49*
validate_shape(
�
+batch_normalization/moving_variance_49/readIdentity&batch_normalization/moving_variance_49*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_49
�
%batch_normalization/FusedBatchNorm_49FusedBatchNormres5b_branch2c!batch_normalization/gamma_49/read batch_normalization/beta_49/read'batch_normalization/moving_mean_49/read+batch_normalization/moving_variance_49/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_49Const*
valueB
 * �:*
dtype0
F
bn5b_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5b_branch2cIdentitybn5b_branch2c*
T0
3
res5bAdd
res5a_reluscale5b_branch2c*
T0
"

res5b_reluRelures5b*
T0
K
res5c_branch2a/kernelConst*
valueB@�*
dtype0
�
res5c_branch2aConv2D
res5b_relures5c_branch2a/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_50/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_50*
dtype0
�
3batch_normalization/gamma_50/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_50*
dtype0
�
-batch_normalization/gamma_50/Initializer/onesFill=batch_normalization/gamma_50/Initializer/ones/shape_as_tensor3batch_normalization/gamma_50/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_50
�
batch_normalization/gamma_50
VariableV2*
shape:�*
shared_name */
_class%
#!loc:@batch_normalization/gamma_50*
dtype0*
	container 
�
#batch_normalization/gamma_50/AssignAssignbatch_normalization/gamma_50-batch_normalization/gamma_50/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_50*
validate_shape(
�
!batch_normalization/gamma_50/readIdentitybatch_normalization/gamma_50*
T0*/
_class%
#!loc:@batch_normalization/gamma_50
�
=batch_normalization/beta_50/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_50*
dtype0
�
3batch_normalization/beta_50/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_50*
dtype0
�
-batch_normalization/beta_50/Initializer/zerosFill=batch_normalization/beta_50/Initializer/zeros/shape_as_tensor3batch_normalization/beta_50/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_50
�
batch_normalization/beta_50
VariableV2*.
_class$
" loc:@batch_normalization/beta_50*
dtype0*
	container *
shape:�*
shared_name 
�
"batch_normalization/beta_50/AssignAssignbatch_normalization/beta_50-batch_normalization/beta_50/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_50*
validate_shape(
�
 batch_normalization/beta_50/readIdentitybatch_normalization/beta_50*
T0*.
_class$
" loc:@batch_normalization/beta_50
�
Dbatch_normalization/moving_mean_50/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_50*
dtype0
�
:batch_normalization/moving_mean_50/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_50*
dtype0
�
4batch_normalization/moving_mean_50/Initializer/zerosFillDbatch_normalization/moving_mean_50/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_50/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_50
�
"batch_normalization/moving_mean_50
VariableV2*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_50*
dtype0*
	container *
shape:�
�
)batch_normalization/moving_mean_50/AssignAssign"batch_normalization/moving_mean_504batch_normalization/moving_mean_50/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_50*
validate_shape(
�
'batch_normalization/moving_mean_50/readIdentity"batch_normalization/moving_mean_50*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_50
�
Gbatch_normalization/moving_variance_50/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_50*
dtype0
�
=batch_normalization/moving_variance_50/Initializer/ones/ConstConst*
dtype0*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_50
�
7batch_normalization/moving_variance_50/Initializer/onesFillGbatch_normalization/moving_variance_50/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_50/Initializer/ones/Const*
T0*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_50
�
&batch_normalization/moving_variance_50
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_50*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_50/AssignAssign&batch_normalization/moving_variance_507batch_normalization/moving_variance_50/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_50*
validate_shape(
�
+batch_normalization/moving_variance_50/readIdentity&batch_normalization/moving_variance_50*9
_class/
-+loc:@batch_normalization/moving_variance_50*
T0
�
%batch_normalization/FusedBatchNorm_50FusedBatchNormres5c_branch2a!batch_normalization/gamma_50/read batch_normalization/beta_50/read'batch_normalization/moving_mean_50/read+batch_normalization/moving_variance_50/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_50Const*
valueB
 * �:*
dtype0
F
bn5c_branch2aIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5c_branch2aIdentitybn5c_branch2a*
T0
6
res5c_branch2a_reluReluscale5c_branch2a*
T0
K
res5c_branch2b/kernelConst*
valueB@�*
dtype0
�
res5c_branch2bConv2Dres5c_branch2a_relures5c_branch2b/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
�
=batch_normalization/gamma_51/Initializer/ones/shape_as_tensorConst*/
_class%
#!loc:@batch_normalization/gamma_51*
valueB:�*
dtype0
�
3batch_normalization/gamma_51/Initializer/ones/ConstConst*/
_class%
#!loc:@batch_normalization/gamma_51*
valueB
 *  �?*
dtype0
�
-batch_normalization/gamma_51/Initializer/onesFill=batch_normalization/gamma_51/Initializer/ones/shape_as_tensor3batch_normalization/gamma_51/Initializer/ones/Const*
T0*/
_class%
#!loc:@batch_normalization/gamma_51*

index_type0
�
batch_normalization/gamma_51
VariableV2*
shared_name */
_class%
#!loc:@batch_normalization/gamma_51*
dtype0*
	container *
shape:�
�
#batch_normalization/gamma_51/AssignAssignbatch_normalization/gamma_51-batch_normalization/gamma_51/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_51*
validate_shape(
�
!batch_normalization/gamma_51/readIdentitybatch_normalization/gamma_51*
T0*/
_class%
#!loc:@batch_normalization/gamma_51
�
=batch_normalization/beta_51/Initializer/zeros/shape_as_tensorConst*.
_class$
" loc:@batch_normalization/beta_51*
valueB:�*
dtype0
�
3batch_normalization/beta_51/Initializer/zeros/ConstConst*.
_class$
" loc:@batch_normalization/beta_51*
valueB
 *    *
dtype0
�
-batch_normalization/beta_51/Initializer/zerosFill=batch_normalization/beta_51/Initializer/zeros/shape_as_tensor3batch_normalization/beta_51/Initializer/zeros/Const*
T0*.
_class$
" loc:@batch_normalization/beta_51*

index_type0
�
batch_normalization/beta_51
VariableV2*
shared_name *.
_class$
" loc:@batch_normalization/beta_51*
dtype0*
	container *
shape:�
�
"batch_normalization/beta_51/AssignAssignbatch_normalization/beta_51-batch_normalization/beta_51/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_51*
validate_shape(
�
 batch_normalization/beta_51/readIdentitybatch_normalization/beta_51*
T0*.
_class$
" loc:@batch_normalization/beta_51
�
Dbatch_normalization/moving_mean_51/Initializer/zeros/shape_as_tensorConst*5
_class+
)'loc:@batch_normalization/moving_mean_51*
valueB:�*
dtype0
�
:batch_normalization/moving_mean_51/Initializer/zeros/ConstConst*5
_class+
)'loc:@batch_normalization/moving_mean_51*
valueB
 *    *
dtype0
�
4batch_normalization/moving_mean_51/Initializer/zerosFillDbatch_normalization/moving_mean_51/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_51/Initializer/zeros/Const*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_51*

index_type0
�
"batch_normalization/moving_mean_51
VariableV2*
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_51*
dtype0*
	container 
�
)batch_normalization/moving_mean_51/AssignAssign"batch_normalization/moving_mean_514batch_normalization/moving_mean_51/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_51*
validate_shape(
�
'batch_normalization/moving_mean_51/readIdentity"batch_normalization/moving_mean_51*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_51
�
Gbatch_normalization/moving_variance_51/Initializer/ones/shape_as_tensorConst*9
_class/
-+loc:@batch_normalization/moving_variance_51*
valueB:�*
dtype0
�
=batch_normalization/moving_variance_51/Initializer/ones/ConstConst*9
_class/
-+loc:@batch_normalization/moving_variance_51*
valueB
 *  �?*
dtype0
�
7batch_normalization/moving_variance_51/Initializer/onesFillGbatch_normalization/moving_variance_51/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_51/Initializer/ones/Const*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_51*

index_type0
�
&batch_normalization/moving_variance_51
VariableV2*9
_class/
-+loc:@batch_normalization/moving_variance_51*
dtype0*
	container *
shape:�*
shared_name 
�
-batch_normalization/moving_variance_51/AssignAssign&batch_normalization/moving_variance_517batch_normalization/moving_variance_51/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_51*
validate_shape(
�
+batch_normalization/moving_variance_51/readIdentity&batch_normalization/moving_variance_51*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_51
�
%batch_normalization/FusedBatchNorm_51FusedBatchNormres5c_branch2b!batch_normalization/gamma_51/read batch_normalization/beta_51/read'batch_normalization/moving_mean_51/read+batch_normalization/moving_variance_51/read*
data_formatNHWC*
is_training( *
epsilon%��'7*
T0
I
batch_normalization/Const_51Const*
dtype0*
valueB
 * �:
F
bn5c_branch2bIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5c_branch2bIdentitybn5c_branch2b*
T0
6
res5c_branch2b_reluReluscale5c_branch2b*
T0
K
res5c_branch2c/kernelConst*
valueB@�*
dtype0
�
res5c_branch2cConv2Dres5c_branch2b_relures5c_branch2c/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
=batch_normalization/gamma_52/Initializer/ones/shape_as_tensorConst*
valueB:�*/
_class%
#!loc:@batch_normalization/gamma_52*
dtype0
�
3batch_normalization/gamma_52/Initializer/ones/ConstConst*
valueB
 *  �?*/
_class%
#!loc:@batch_normalization/gamma_52*
dtype0
�
-batch_normalization/gamma_52/Initializer/onesFill=batch_normalization/gamma_52/Initializer/ones/shape_as_tensor3batch_normalization/gamma_52/Initializer/ones/Const*
T0*

index_type0*/
_class%
#!loc:@batch_normalization/gamma_52
�
batch_normalization/gamma_52
VariableV2*/
_class%
#!loc:@batch_normalization/gamma_52*
dtype0*
	container *
shape:�*
shared_name 
�
#batch_normalization/gamma_52/AssignAssignbatch_normalization/gamma_52-batch_normalization/gamma_52/Initializer/ones*
use_locking(*
T0*/
_class%
#!loc:@batch_normalization/gamma_52*
validate_shape(
�
!batch_normalization/gamma_52/readIdentitybatch_normalization/gamma_52*
T0*/
_class%
#!loc:@batch_normalization/gamma_52
�
=batch_normalization/beta_52/Initializer/zeros/shape_as_tensorConst*
valueB:�*.
_class$
" loc:@batch_normalization/beta_52*
dtype0
�
3batch_normalization/beta_52/Initializer/zeros/ConstConst*
valueB
 *    *.
_class$
" loc:@batch_normalization/beta_52*
dtype0
�
-batch_normalization/beta_52/Initializer/zerosFill=batch_normalization/beta_52/Initializer/zeros/shape_as_tensor3batch_normalization/beta_52/Initializer/zeros/Const*
T0*

index_type0*.
_class$
" loc:@batch_normalization/beta_52
�
batch_normalization/beta_52
VariableV2*
	container *
shape:�*
shared_name *.
_class$
" loc:@batch_normalization/beta_52*
dtype0
�
"batch_normalization/beta_52/AssignAssignbatch_normalization/beta_52-batch_normalization/beta_52/Initializer/zeros*
use_locking(*
T0*.
_class$
" loc:@batch_normalization/beta_52*
validate_shape(
�
 batch_normalization/beta_52/readIdentitybatch_normalization/beta_52*
T0*.
_class$
" loc:@batch_normalization/beta_52
�
Dbatch_normalization/moving_mean_52/Initializer/zeros/shape_as_tensorConst*
valueB:�*5
_class+
)'loc:@batch_normalization/moving_mean_52*
dtype0
�
:batch_normalization/moving_mean_52/Initializer/zeros/ConstConst*
valueB
 *    *5
_class+
)'loc:@batch_normalization/moving_mean_52*
dtype0
�
4batch_normalization/moving_mean_52/Initializer/zerosFillDbatch_normalization/moving_mean_52/Initializer/zeros/shape_as_tensor:batch_normalization/moving_mean_52/Initializer/zeros/Const*
T0*

index_type0*5
_class+
)'loc:@batch_normalization/moving_mean_52
�
"batch_normalization/moving_mean_52
VariableV2*
	container *
shape:�*
shared_name *5
_class+
)'loc:@batch_normalization/moving_mean_52*
dtype0
�
)batch_normalization/moving_mean_52/AssignAssign"batch_normalization/moving_mean_524batch_normalization/moving_mean_52/Initializer/zeros*
use_locking(*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_52*
validate_shape(
�
'batch_normalization/moving_mean_52/readIdentity"batch_normalization/moving_mean_52*
T0*5
_class+
)'loc:@batch_normalization/moving_mean_52
�
Gbatch_normalization/moving_variance_52/Initializer/ones/shape_as_tensorConst*
valueB:�*9
_class/
-+loc:@batch_normalization/moving_variance_52*
dtype0
�
=batch_normalization/moving_variance_52/Initializer/ones/ConstConst*
valueB
 *  �?*9
_class/
-+loc:@batch_normalization/moving_variance_52*
dtype0
�
7batch_normalization/moving_variance_52/Initializer/onesFillGbatch_normalization/moving_variance_52/Initializer/ones/shape_as_tensor=batch_normalization/moving_variance_52/Initializer/ones/Const*

index_type0*9
_class/
-+loc:@batch_normalization/moving_variance_52*
T0
�
&batch_normalization/moving_variance_52
VariableV2*
	container *
shape:�*
shared_name *9
_class/
-+loc:@batch_normalization/moving_variance_52*
dtype0
�
-batch_normalization/moving_variance_52/AssignAssign&batch_normalization/moving_variance_527batch_normalization/moving_variance_52/Initializer/ones*
use_locking(*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_52*
validate_shape(
�
+batch_normalization/moving_variance_52/readIdentity&batch_normalization/moving_variance_52*
T0*9
_class/
-+loc:@batch_normalization/moving_variance_52
�
%batch_normalization/FusedBatchNorm_52FusedBatchNormres5c_branch2c!batch_normalization/gamma_52/read batch_normalization/beta_52/read'batch_normalization/moving_mean_52/read+batch_normalization/moving_variance_52/read*
epsilon%��'7*
T0*
data_formatNHWC*
is_training( 
I
batch_normalization/Const_52Const*
valueB
 * �:*
dtype0
F
bn5c_branch2cIdentity"batch_normalization/FusedBatchNorm*
T0
4
scale5c_branch2cIdentitybn5c_branch2c*
T0
3
res5cAdd
res5b_reluscale5c_branch2c*
T0
"

res5c_reluRelures5c*
T0
I
rpn_conv/3x3/kernelConst*
valueB@�*
dtype0
�
rpn_conv/3x3Conv2D
res4f_relurpn_conv/3x3/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
+
rpn_relu/3x3Relurpn_conv/3x3*
T0
J
rpn_cls_score/kernelConst*
valueB�*
dtype0
�
rpn_cls_scoreConv2Drpn_relu/3x3rpn_cls_score/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
J
rpn_bbox_pred/kernelConst*
valueB�$*
dtype0
�
rpn_bbox_predConv2Drpn_relu/3x3rpn_bbox_pred/kernel*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
c
rpn_cls_score_reshapeReshaperpn_cls_scorerpn_cls_score_reshape/shape*
T0*
Tshape0
X
rpn_cls_score_reshape/shapeConst*%
valueB"   �����      *
dtype0
7
rpn_cls_probSoftmaxrpn_cls_score_reshape*
T0
`
rpn_cls_prob_reshapeReshaperpn_cls_probrpn_cls_prob_reshape/shape*
T0*
Tshape0
W
rpn_cls_prob_reshape/shapeConst*%
valueB"   �����      *
dtype0
G
conv_new_1/kernelConst*
valueB@�*
dtype0
�

conv_new_1Conv2D
res5c_reluconv_new_1/kernel*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

,
conv_new_1_reluRelu
conv_new_1*
T0
F
rfcn_cls/kernelConst*
valueB��*
dtype0
�
rfcn_clsConv2Dconv_new_1_relurfcn_cls/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
G
rfcn_bbox/kernelConst*
dtype0*
valueB��
�
	rfcn_bboxConv2Dconv_new_1_relurfcn_bbox/kernel*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID" 