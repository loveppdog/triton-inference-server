// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>
#include <mutex>
#include "src/backends/backend/examples/backend_input_collector.h"
#include "src/backends/backend/examples/backend_model.h"
#include "src/backends/backend/examples/backend_model_instance.h"
#include "src/backends/backend/examples/backend_output_responder.h"
#include "src/backends/backend/examples/backend_utils.h"
#include "src/backends/backend/onnxruntime/loader.h"
#include "src/backends/backend/onnxruntime/onnx_utils.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_provider_factory.h>
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
#include <tensorrt_provider_factory.h>
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT

#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
#include <openvino_provider_factory.h>
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO

//
// ONNX Runtime Backend that implements the TRITONBACKEND API.
//

namespace ni = nvidia::inferenceserver;
namespace nib = nvidia::inferenceserver::backend;

namespace triton { namespace backend { namespace onnxruntime {

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public nib::BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  OrtSessionOptions* SessionOptions() { return session_options_.get(); }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Session options used when creating a ORT session.
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> session_options_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const nib::BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : nib::BackendModel(triton_model)
{
  // Create session options that will be cloned and used for each
  // instance when creating that instance's session.
  OrtSessionOptions* soptions;
  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->CreateSessionOptions(&soptions));
  session_options_.reset(soptions);

  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->SetIntraOpNumThreads(soptions, 1));

  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  {
    ni::TritonJson::Value optimization;
    if (ModelConfig().Find("optimization", &optimization)) {
      ni::TritonJson::Value graph;
      if (optimization.Find("graph", &graph)) {
        int64_t graph_level = 0;
        THROW_IF_BACKEND_MODEL_ERROR(graph.MemberAsInt("level", &graph_level));
        if (graph_level == -1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        } else if (graph_level == 1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        }
      }
    }
  }
  THROW_IF_BACKEND_MODEL_ORT_ERROR(
      ort_api->SetSessionGraphOptimizationLevel(soptions, optimization_level));

  // FIXME. Is it possible to share a single OrtSession across
  // multiple instances? If so then should move loading and validation
  // of the session to here instead of creating a session for each
  // instance in ModelStateInstance::Create().
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public nib::BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      ni::TritonJson::Value& sequence_batching, const std::string& control_kind,
      bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      ni::TritonJson::Value& sequence_batching, const std::string& control_kind,
      bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();

  ModelState* model_state_;

  // The full path to the ONNX model file.
  std::string model_path_;

  // Onnx Runtime variables that are used across runs on this
  // instance.
  OrtSession* session_;
  OrtAllocator* allocator_;

  // Onnx Runtime variables that will be reset and used for every run
  // on this instance.
  std::vector<OrtValue*> input_tensors_;
  std::vector<OrtValue*> output_tensors_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const nib::BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), session_(nullptr), allocator_(nullptr)
{
  // Find the ONNX file that describes the model itself. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.onnx").
  std::string cc_model_filename = ArtifactFilename();
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.onnx";
  }

  model_path_ =
      nib::JoinPath({model_state->RepositoryPath(),
                std::to_string(model_state->Version()), cc_model_filename});

  // If the model path is a directory then the actual model is
  // <dir>/model.onnx.
  {
    bool is_dir;
    THROW_IF_BACKEND_MODEL_ERROR(nib::IsDirectory(model_path_, &is_dir));
    if (is_dir) {
      model_path_ = nib::JoinPath({model_path_, "model.onnx"});
    }
  }

  {
    bool exists;
    THROW_IF_BACKEND_MODEL_ERROR(nib::FileExists(model_path_, &exists));
    if (!exists) {
      throw nib::BackendModelInstanceException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          (std::string("unable to find '") + model_path_ +
           "' for model instance '" + Name() + "'")
              .c_str()));
    }
  }

  // Make a clone for the session options for this instance...
  OrtSessionOptions* soptions;
  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->CloneSessionOptions(model_state->SessionOptions(), &soptions));
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> soptions_wrapper(
      soptions);

  ni::TritonJson::Value& model_config = model_state->ModelConfig();

  // Set GPU execution execution providers
  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (model_state->ModelConfig()
            .optimization()
            .has_execution_accelerators()) {
      // Don't need to ensure uniqueness of the providers, ONNX
      // Runtime will check it.
      for (const auto& execution_accelerator :
           model_state->ModelConfig()
               .optimization()
               .execution_accelerators()
               .gpu_execution_accelerator()) {
#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
        if (execution_accelerator.name() == kTensorRTExecutionAccelerator) {
          THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
              OrtSessionOptionsAppendExecutionProvider_Tensorrt(
                  soptions, DeviceId()));
          LOG_MESSAGE(
              TRITONSERVER_LOG_VERBOSE,
              (std::string("TensorRT Execution Accelerator is set for '") +
               instance_name + "' on device " + std::to_string(DeviceId()))
                  .c_str());
        } else
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT
        {
          throw new nib::BackendModelInstanceException(TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unknown Execution Accelerator '") +
               execution_accelerator.name() + "' is requested")
                  .c_str()));
        }
      }
    }
    THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
        OrtSessionOptionsAppendExecutionProvider_CUDA(soptions, gpu_device));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("CUDA Execution Accelerator is set for '") +
         instance_name + "' on device " + std::to_string(DeviceId()))
            .c_str());
#else
    throw new nib::BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "GPU instances not supported"));
#endif  // TRITON_ENABLE_GPU
  }

  // Check for an enable OpenVINO execution accelerator
  bool need_lock = false;
  if (model_state->ModelConfig().optimization().has_execution_accelerators()) {
    for (const auto& execution_accelerator : model_state->ModelConfig()
                                                 .optimization()
                                                 .execution_accelerators()
                                                 .cpu_execution_accelerator()) {
      if (execution_accelerator.name() == kOpenVINOExecutionAccelerator) {
#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
        need_lock = true;
        THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
            OrtSessionOptionsAppendExecutionProvider_OpenVINO(soptions, ""));
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("OpenVINO Execution Accelerator is set for '") +
             instance_name + "' on CPU")
                .c_str());
#else
        throw nib::BackendModelInstanceException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "OpenVINO Execution Accelerator is not enabled"));
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO
      } else {
        throw nib::BackendModelInstanceException(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unknown Execution Accelerator '") +
             execution_accelerator.name() + "' is requested")
                .c_str()));
      }
    }
  }

  // ONNX session creation with OpenVINO is not thread-safe,
  // so multiple creations are serialized with a global lock.
  static std::mutex global_context_mu;
  std::unique_lock<std::mutex> glock(global_context_mu, std::defer_lock);
  if (need_lock) {
    glock.lock();
  }

  // Register all op libraries that contain custom operations.
  if (model_state->ModelConfig().has_model_operations()) {
    auto model_ops = model_state->ModelConfig().model_operations();
    for (const auto& lib_filename : model_ops.op_library_filename()) {
      void* library_handle = nullptr;  // leak this, no harm.
      THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->RegisterCustomOpsLibrary(
          soptions, lib_filename.c_str(), &library_handle));
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(OnnxLoader::LoadSession(
      true /* is_path */, model_path_, soptions, &session_));
  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->GetAllocatorWithDefaultOptions(&allocator_));

  size_t expected_input_cnt = (size_t)Config().input().size();

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  ni::TritonJson::Value sequence_batching;
  if (model_config.Find("sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  ReleaseOrtRunResources();
  if (session_ != nullptr) {
    OnnxLoader::UnloadSession(session_);
  }
  // 'allocator_' is default allocator which is managed by ONNX Runtime
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    ni::TritonJson::Value& sequence_batching, const std::string& control_kind,
    bool required, bool* have_control)
{
  std::string tensor_name;
  inference::DataType tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      batcher, model_name, control_kind, required, &tensor_name,
      &tensor_datatype, nullptr, nullptr, nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (max_batch_size_ > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_name +
           "', sequence control '" + tensor_name + "' in model has dims " +
           nib::ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ConvertToOnnxDataType(tensor_datatype) != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_name +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           inference::DataType_Name(tensor_datatype))
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    ni::TritonJson::Value& sequence_batching, const std::string& control_kind,
    bool required, bool* have_control)
{
  std::string tensor_name;
  inference::DataType tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      batcher, model_name, control_kind, required, &tensor_name,
      &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return Status(
          Status::Code::INTERNAL,
          "configuration specified sequence control '" + tensor_name +
              "', but model does not provide that input");
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (max_batch_size_ > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "' in model has dims " +
              DimsListToString(debatched_dims) + " but dims [1] is expected");
    }

    if (ConvertToOnnxDataType(tensor_datatype) != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + "', sequence control '" +
              tensor_name + "', the model expects data-type " +
              OnnxDataTypeName(iit->second.type_) +
              " but the model configuration specifies data-type " +
              inference::DataType_Name(tensor_datatype));
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  RETURN_IF_ERROR(InputNames(session_, input_tensor_names));

  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(InputInfos(session_, allocator_, input_tensor_infos));

  if (input_tensor_infos.size() != expected_input_cnt) {
    return Status(
        Status::Code::INVALID_ARG,
        "unable to load model '" + model_name + "', configuration expects " +
            std::to_string(expected_input_cnt) + " inputs, model provides " +
            std::to_string(input_tensor_infos.size()));
  }

  for (const auto& io : ios) {
    auto iit = input_tensor_infos.find(io.name());
    if (iit == input_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, input_tensor_names));
    }

    auto onnx_data_type = ConvertToOnnxDataType(io.data_type());
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + inference::DataType_Name(io.data_type()) +
              " for input '" + io.name() + "' for model '" + model_name + "'");
    } else if (onnx_data_type != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + ", unexpected datatype " +
              inference::DataType_Name(
                  ConvertFromOnnxDataType(iit->second.type_)) +
              " for input '" + io.name() + "', expecting " +
              inference::DataType_Name(io.data_type()));
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(CompareDimsSupported(
        model_name, io.name(), iit->second.dims_, dims, max_batch_size_,
        false /* compare_exact */));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  std::set<std::string> output_tensor_names;
  RETURN_IF_ERROR(OutputNames(session_, output_tensor_names));

  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(OutputInfos(session_, allocator_, output_tensor_infos));

  for (const auto& io : ios) {
    auto iit = output_tensor_infos.find(io.name());
    if (iit == output_tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto onnx_data_type = ConvertToOnnxDataType(io.data_type());
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return Status(
          Status::Code::INTERNAL,
          "unsupported datatype " + inference::DataType_Name(io.data_type()) +
              " for output '" + io.name() + "' for model '" + model_name + "'");
    } else if (onnx_data_type != iit->second.type_) {
      return Status(
          Status::Code::INVALID_ARG,
          "unable to load model '" + model_name + ", unexpected datatype " +
              inference::DataType_Name(
                  ConvertFromOnnxDataType(iit->second.type_)) +
              " for output '" + io.name() + "', expecting " +
              inference::DataType_Name(io.data_type()));
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    const DimsList& dims =
        (io.has_reshape()) ? io.reshape().shape() : io.dims();
    RETURN_IF_ERROR(CompareDimsSupported(
        model_name, io.name(), iit->second.dims_, dims, max_batch_size_,
        true /* compare_exact */));
  }

  return nullptr;  // success
}

void
OnnxBackend::Context::Run(
    InferenceBackend* base,
    std::vector<std::unique_ptr<InferenceRequest>>&& requests)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << requests.size()
                 << " request requests";

  INFER_STATS_DECL_TIMESTAMP(compute_start_ns);

  // For each request in 'requests' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each requests
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& request : requests) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (request == nullptr) {
      InferenceRequest::RespondIfError(
          requests,
          Status(
              Status::Code::INTERNAL,
              "null request given to TensorFlow runner for '" + name_ + "'"),
          true /* release_requests */);
      return;
    }

    total_batch_size += std::max(1U, request->BatchSize());
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return;
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    InferenceRequest::RespondIfError(
        requests,
        Status(
            Status::Code::INTERNAL,
            "dynamic batch size " + std::to_string(total_batch_size) +
                " for '" + name_ + "', max allowed is " +
                std::to_string(max_batch_size_)),
        true /* release_requests */);
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<std::unique_ptr<InferenceResponse>> responses;
  responses.reserve(requests.size());

  for (auto& request : requests) {
    std::unique_ptr<InferenceResponse> response;
    Status status = request->ResponseFactory().CreateResponse(&response);
    if (!status.IsOk()) {
      InferenceRequest::RespondIfError(request, status);
      response.reset();
    }

    responses.emplace_back(std::move(response));
  }

  // Use scoped class to clean up ORT tensors
  struct ScopedCleanup {
    ScopedCleanup(Context* ctx) : ctx_(ctx) {}
    ~ScopedCleanup()
    {
      if (ctx_ != nullptr) {
        ctx_->ReleaseOrtRunResources();
      }
    }
    Context* ctx_;
  } io_tensor_wrapper(this);

  // Hold reference to each buffer of input data so that it stays
  // until the inference has completed.
  std::vector<std::unique_ptr<AllocatedMemory>> input_buffers;
  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, &responses, enable_pinned_input_, stream_);
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      SetInputTensors(
          total_batch_size, requests, &responses, &collector, &input_buffers,
          &input_names, &cuda_copy),
      "error sending ONNX response");

  // Request to retrieve all output specified in model config
  // and reserve placeholder for output tensors
  std::vector<const char*> output_names;
  for (const auto& output : base->Config().output()) {
    output_names.emplace_back(output.name().c_str());
    output_tensors_.emplace_back(nullptr);
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif

  INFER_STATS_DECL_TIMESTAMP(compute_input_end_ns);

  // Run...
  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      OrtRun(input_names, output_names), "error sending ONNX response");

  INFER_STATS_DECL_TIMESTAMP(compute_output_start_ns);

  FAIL_ALL_AND_RETURN_IF_ERROR(
      requests, responses, metric_reporter_.get(),
      ReadOutputTensors(total_batch_size, output_names, requests, &responses),
      "error sending ONNX response");

#ifdef TRITON_ENABLE_STATS
  INFER_STATS_DECL_TIMESTAMP(compute_end_ns);

  // Report stats and trace
  for (size_t i = 0; i < requests.size(); ++i) {
    auto& request = requests[i];
    request->ReportStatistics(
        metric_reporter_.get(), (responses[i] != nullptr), compute_start_ns,
        compute_input_end_ns, compute_output_start_ns, compute_end_ns);

#ifdef TRITON_ENABLE_TRACING
    if (request->Trace() != nullptr) {
      auto& trace = request->Trace();
      trace->Report(TRITONSERVER_TRACE_COMPUTE_START, compute_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_INPUT_END, compute_input_end_ns);
      trace->Report(
          TRITONSERVER_TRACE_COMPUTE_OUTPUT_START, compute_output_start_ns);
      trace->Report(TRITONSERVER_TRACE_COMPUTE_END, compute_end_ns);
    }
#endif  // TRITON_ENABLE_TRACING
  }

  // Also reporting batch stats
  base->MutableStatsAggregator()->UpdateInferBatchStats(
      metric_reporter_.get(), total_batch_size, compute_start_ns,
      compute_input_end_ns, compute_output_start_ns, compute_end_ns);
#endif  // TRITON_ENABLE_STATS

  // Send all the responses that haven't already been sent because of
  // an earlier error.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_STATUS_ERROR(
          InferenceResponse::Send(
              std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL),
          "failed to send TensorFlow backend response");
    }
  }

  // Release all requests.
  for (auto& request : requests) {
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);
  }
}

Status
OnnxBackend::Context::SetInputTensors(
    size_t total_batch_size,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    BackendInputCollector* collector,
    std::vector<std::unique_ptr<AllocatedMemory>>* input_buffers,
    std::vector<const char*>* input_names, bool* cuda_copy)
{
  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  for (const auto& pr : requests[0]->ImmutableInputs()) {
    const auto& name = pr.first;
    const auto& repr_input = pr.second;
    const auto& batch1_shape = repr_input->Shape();

    input_names->emplace_back(name.c_str());
    input_tensors_.emplace_back(nullptr);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape;
    batchn_shape.reserve(batch1_shape.size() + 1);
    if (max_batch_size_ != NO_BATCHING) {
      batchn_shape.push_back(total_batch_size);
    }
    batchn_shape.insert(
        batchn_shape.end(), batch1_shape.begin(), batch1_shape.end());

    const inference::DataType datatype = repr_input->DType();

    // [TODO] currently ONNX Runtime only recognize input data on CPU
    // https://github.com/microsoft/onnxruntime/issues/1621
    if (datatype != inference::DataType::TYPE_STRING) {
      input_buffers->emplace_back(new AllocatedMemory(
          GetByteSize(datatype, batchn_shape), TRITONSERVER_MEMORY_CPU_PINNED,
          0));
      TRITONSERVER_MemoryType mem_type;
      auto input_buffer = input_buffers->back()->MutableBuffer(&mem_type);
      auto total_byte_size = input_buffers->back()->TotalByteSize();

      // Create ORT Tensor
      const OrtMemoryInfo* allocator_info;
      RETURN_IF_ORT_ERROR(
          ort_api->AllocatorGetInfo(allocator_, &allocator_info));
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          allocator_info, (void*)input_buffer, total_byte_size,
          batchn_shape.data(), batchn_shape.size(),
          ConvertToOnnxDataType(datatype), &input_tensors_.back()));

      collector->ProcessTensor(
          name, datatype, batch1_shape, input_buffer, total_byte_size, mem_type,
          0);
    } else {
      // For String input, we need to obtain tensor info differently
      size_t total_byte_size = 0;
      std::vector<size_t> expected_byte_sizes;
      std::vector<size_t> expected_element_cnts;
      expected_byte_sizes.reserve(requests.size());
      expected_element_cnts.reserve(requests.size());
      for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
        const InferenceRequest::Input* in;
        auto status = requests[ridx]->ImmutableInput(name, &in);
        // Skip input in this request if failed to retrieve it
        if (!status.IsOk()) {
          if ((*responses)[ridx] != nullptr) {
            InferenceResponse::SendWithStatus(
                std::move((*responses)[ridx]),
                TRITONSERVER_RESPONSE_COMPLETE_FINAL, status);
          }
          expected_byte_sizes.push_back(0);
          expected_element_cnts.push_back(0);
        } else {
          expected_element_cnts.push_back(
              GetElementCount(in->ShapeWithBatchDim()));
          expected_byte_sizes.push_back(in->Data()->TotalByteSize());
        }
        total_byte_size += expected_byte_sizes.back();
      }
      // For string input, the copy to contiguous buffer is needed because ORT
      // expects elements to be C strings thus we need to modify input buffer.
      // Reserve one more byte at the end of input_buffer to ensure last
      // element of String data can become valid C string.
      input_buffers->emplace_back(new AllocatedMemory(
          total_byte_size + 1, TRITONSERVER_MEMORY_CPU_PINNED, 0));
      TRITONSERVER_MemoryType mem_type;
      auto input_buffer = input_buffers->back()->MutableBuffer(&mem_type);
      size_t buffer_offset = 0;
      bool string_cuda_copy = false;
      for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
        const InferenceRequest::Input* in;
        auto status = requests[ridx]->ImmutableInput(name, &in);
        if (status.IsOk() && ((*responses)[ridx] != nullptr)) {
          const void* src_buffer;
          size_t src_byte_size;
          TRITONSERVER_MemoryType src_memory_type;
          int64_t src_memory_type_id;
          size_t input_offset = 0;
          for (size_t idx = 0; idx < in->DataBufferCount(); ++idx) {
            status = in->DataBuffer(
                idx, &src_buffer, &src_byte_size, &src_memory_type,
                &src_memory_type_id);
            if (status.IsOk()) {
              if (input_offset + src_byte_size > expected_byte_sizes[ridx]) {
                status = Status(
                    Status::Code::INVALID_ARG,
                    "buffer size for input '" + name +
                        "' exceeds batch byte size " +
                        std::to_string(expected_byte_sizes[ridx]));
              } else {
                bool cuda_used = false;
                status = CopyBuffer(
                    name, src_memory_type, src_memory_type_id, mem_type, 0,
                    src_byte_size, src_buffer,
                    input_buffer + buffer_offset + input_offset, stream_,
                    &cuda_used);
                string_cuda_copy |= cuda_used;
              }
            }
            if (status.IsOk()) {
              input_offset += src_byte_size;
            } else {
              break;
            }
          }
        }
        if (!status.IsOk() && ((*responses)[ridx] != nullptr)) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[ridx]),
              TRITONSERVER_RESPONSE_COMPLETE_FINAL, status);
        }
        buffer_offset += expected_byte_sizes[ridx];
      }

#ifdef TRITON_ENABLE_GPU
      // Synchronize to ensure the buffer is ready to be modified
      if (string_cuda_copy) {
        cudaStreamSynchronize(stream_);
      }
#endif  // TRITON_ENABLE_GPU

      std::vector<const char*> string_data;
      // Modify input buffer and set string expected by ORT
      SetStringInputBuffer(
          name, expected_byte_sizes, expected_element_cnts, responses,
          input_buffer, &string_data);
      input_buffer[total_byte_size] = 0;

      RETURN_IF_ORT_ERROR(ort_api->CreateTensorAsOrtValue(
          allocator_, batchn_shape.data(), batchn_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RETURN_IF_ORT_ERROR(ort_api->FillStringTensor(
          input_tensors_.back(), string_data.data(), string_data.size()));
    }
  }
  // Finalize...
  *cuda_copy |= collector->Finalize();
  return Status::Success;
}

Status
OnnxBackend::Context::OrtRun(
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names)
{
  RETURN_IF_ORT_ERROR(ort_api->Run(
      session_, NULL /* run options */, input_names.data(),
      (const OrtValue* const*)input_tensors_.data(), input_tensors_.size(),
      output_names.data(), output_names.size(), output_tensors_.data()));
  return Status::Success;
}

void
OnnxBackend::Context::SetStringInputBuffer(
    const std::string& name, const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<std::unique_ptr<InferenceResponse>>* responses,
    char* input_buffer, std::vector<const char*>* string_data)
{
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    size_t element_cnt = 0;
    if ((*responses)[idx] != nullptr) {
      size_t remaining_bytes = expected_byte_size;
      char* data_content = input_buffer + buffer_copy_offset;
      // Continue if the remaining bytes may still contain size info
      while (remaining_bytes >= sizeof(uint32_t)) {
        if (element_cnt >= expected_element_cnt) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[idx]),
              TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              Status(
                  Status::Code::INVALID_ARG,
                  "unexpected number of string elements " +
                      std::to_string(element_cnt + 1) +
                      " for inference input '" + name + "', expecting " +
                      std::to_string(expected_element_cnt)));
          break;
        }

        const uint32_t len = *(reinterpret_cast<const uint32_t*>(data_content));
        remaining_bytes -= sizeof(uint32_t);
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *data_content = 0;
        data_content = data_content + sizeof(uint32_t);
        if (len > remaining_bytes) {
          InferenceResponse::SendWithStatus(
              std::move((*responses)[idx]),
              TRITONSERVER_RESPONSE_COMPLETE_FINAL,
              Status(
                  Status::Code::INVALID_ARG,
                  "incomplete string data for inference input '" + name +
                      "', expecting string of length " + std::to_string(len) +
                      " but only " + std::to_string(remaining_bytes) +
                      " bytes available"));
          break;
        } else {
          string_data->push_back(data_content);
          element_cnt++;
          data_content = data_content + len;
          remaining_bytes -= len;
        }
      }
    }

    FillStringData(string_data, expected_element_cnt - element_cnt);

    buffer_copy_offset += expected_byte_size;
  }
}

void
OnnxBackend::Context::FillStringData(
    std::vector<const char*>* string_data, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_data->push_back(empty);
  }
}

Status
OnnxBackend::Context::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses)
{
  BackendResponder responder(
      requests, responses, max_batch_size_, enable_pinned_output_, stream_);

  // Use to hold string output contents
  bool cuda_copy = false;
  std::vector<std::vector<char>> string_buffers;
  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = std::string(output_names[idx]);

    OrtValue* output_tensor = output_tensors_[idx];
    if (output_tensor == nullptr) {
      return Status(
          Status::Code::INTERNAL,
          "output tensor '" + name + "' does not found");
    }

    // Get output type and shape
    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(ort_api->GetTypeInfo(output_tensor, &typeinfo));
    std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

    const OrtTensorTypeAndShapeInfo* type_and_shape;
    RETURN_IF_ORT_ERROR(
        ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

    size_t num_dims;
    RETURN_IF_ORT_ERROR(ort_api->GetDimensionsCount(type_and_shape, &num_dims));

    std::vector<int64_t> batchn_shape(num_dims);
    RETURN_IF_ORT_ERROR(ort_api->GetDimensions(
        type_and_shape, batchn_shape.data(), batchn_shape.size()));
    const size_t element_count = GetElementCount(batchn_shape);

    ONNXTensorElementDataType type;
    RETURN_IF_ORT_ERROR(ort_api->GetTensorElementType(type_and_shape, &type));

    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      const size_t batch1_element_cnt = element_count / total_batch_size;
      size_t total_length = 0;
      RETURN_IF_ORT_ERROR(
          ort_api->GetStringTensorDataLength(output_tensor, &total_length));

      string_buffers.emplace_back(std::vector<char>(total_length));
      auto content = string_buffers.back().data();
      size_t offsets[element_count + 1];
      RETURN_IF_ORT_ERROR(ort_api->GetStringTensorContent(
          output_tensor, content, total_length, offsets, element_count));
      // Mark "passed end byte offset"
      offsets[element_count] = total_length;

      cuda_copy |= SetStringOutputBuffer(
          name, batch1_element_cnt, content, offsets, &batchn_shape, requests,
          responses);
    } else {
      // Fixed size data type...
      char* output_buffer = nullptr;
      RETURN_IF_ORT_ERROR(
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));

      // [TODO] currently ONNX output data are always on CPU
      // https://github.com/microsoft/onnxruntime/issues/1621
      responder.ProcessTensor(
          name, ConvertFromOnnxDataType(type), batchn_shape, output_buffer,
          TRITONSERVER_MEMORY_CPU, 0);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

bool
OnnxBackend::Context::SetStringOutputBuffer(
    const std::string& name, const size_t batch1_element_cnt,
    const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape,
    const std::vector<std::unique_ptr<InferenceRequest>>& requests,
    std::vector<std::unique_ptr<InferenceResponse>>* responses)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < requests.size(); ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];
    const size_t expected_element_cnt =
        std::max(1U, request->BatchSize()) * batch1_element_cnt;

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just
    // skip it in the 'content'.
    if ((response != nullptr) &&
        (request->ImmutableRequestedOutputs().find(name) !=
         request->ImmutableRequestedOutputs().end())) {
      if (max_batch_size_ != NO_BATCHING) {
        (*batchn_shape)[0] = request->BatchSize();
      }
      InferenceResponse::Output* response_output = nullptr;
      response->AddOutput(
          name, inference::DataType::TYPE_STRING, *batchn_shape,
          &response_output);
      // Calculate expected byte size in advance using string offsets
      const size_t data_byte_size =
          offsets[element_idx + expected_element_cnt] - offsets[element_idx];
      const size_t expected_byte_size =
          data_byte_size + sizeof(uint32_t) * expected_element_cnt;

      void* buffer;
      TRITONSERVER_MemoryType actual_memory_type =
          TRITONSERVER_MEMORY_CPU_PINNED;
      int64_t actual_memory_type_id = 0;
      Status status = response_output->AllocateDataBuffer(
          &buffer, expected_byte_size, &actual_memory_type,
          &actual_memory_type_id);
      if (status.IsOk()) {
        bool cuda_used = false;
        size_t copied_byte_size = 0;
        for (size_t e = 0; e < expected_element_cnt; ++e) {
          const uint32_t len =
              offsets[element_idx + e + 1] - offsets[element_idx + e];
          // Prepend size of the string
          status = CopyBuffer(
              name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
              0 /* src_memory_type_id */, actual_memory_type,
              actual_memory_type_id, sizeof(uint32_t),
              static_cast<const void*>(&len),
              static_cast<char*>(buffer) + copied_byte_size, stream_,
              &cuda_used);
          if (!status.IsOk()) {
            break;
          }

          cuda_copy |= cuda_used;
          copied_byte_size += sizeof(uint32_t);

          // Copy raw string content
          status = CopyBuffer(
              name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
              0 /* src_memory_type_id */, actual_memory_type,
              actual_memory_type_id, len, content + offsets[element_idx + e],
              static_cast<char*>(buffer) + copied_byte_size, stream_,
              &cuda_used);
          if (!status.IsOk()) {
            break;
          }

          cuda_copy |= cuda_used;
          copied_byte_size += len;
        }
      }
      if (!status.IsOk()) {
        InferenceResponse::SendWithStatus(
            std::move(response), TRITONSERVER_RESPONSE_COMPLETE_FINAL, status);
      }
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}

void
OnnxBackend::Context::ReleaseOrtRunResources()
{
  // Release input tensor if set
  for (auto& tensor : input_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  input_tensors_.clear();

  // Release output tensor if set
  for (auto& tensor : output_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  output_tensors_.clear();
}

std::ostream&
operator<<(std::ostream& out, const OnnxBackend& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context->name_ << ", gpu="
        << ((context->gpu_device_ == OnnxBackend::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context->gpu_device_))
        << ", max_batch_size="
        << ((context->max_batch_size_ == OnnxBackend::Context::NO_BATCHING)
                ? "<none>"
                : std::to_string(context->max_batch_size_))
        << std::endl;
  }

  return out;
}

}}}  // namespace triton::backend::onnxruntime

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // Onetime initialization for the onnxruntime loader.
  RETURN_IF_ERROR(OnnxLoader::Init());

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  LOG_IF_ERROR(OnnxLoader::Stop(), "failed to stop OnnxLoader");
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ExecuteRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
