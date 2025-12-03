#include "ClientRunningRequest.h"
#include "ClientRunner.hpp"
#include "auto/tl/cocoon_api.h"
#include "checksum.h"
#include "nlohmann/detail/conversions/from_json.hpp"
#include "nlohmann/detail/conversions/to_json.hpp"
#include "td/actor/actor.h"
#include "td/utils/StringBuilder.h"
#include "td/utils/buffer.h"
#include "runners/helpers/HttpSender.hpp"

#include "auto/tl/cocoon_api_json.h"
#include "cocoon-tl-utils/cocoon-tl-utils.hpp"
#include "td/utils/port/Clocks.h"
#include "tl/TlObject.h"
#include <utility>

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

namespace cocoon {

void ClientRunningRequest::start_up() {
  LOG(INFO) << "client request " << request_id_.to_hex() << ": starting";

  keep_alive_ = in_request_->keep_alive();

  stats()->requests_received++;
  auto P = td::PromiseCreator::lambda([self_id = actor_id(this)](td::Result<td::BufferSlice> R) mutable {
    if (R.is_ok()) {
      td::actor::send_closure(self_id, &ClientRunningRequest::on_payload_downloaded, R.move_as_ok());
    } else {
      td::actor::send_closure(self_id, &ClientRunningRequest::return_error, R.move_as_error(), nullptr);
    }
  });

  td::actor::create_actor<HttpPayloadReceiver>("payloadreceiver", std::move(in_payload_), std::move(P)).release();
}

void ClientRunningRequest::on_payload_downloaded(td::BufferSlice payload) {
  LOG(DEBUG) << "client request " << request_id_.to_hex() << ": downloaded payload, size=" << payload.size();
  stats()->request_bytes_received += (double)payload.size();
  std::string model = "test1";
  td::int32 max_coefficient = 4000;
  td::int32 max_tokens = 1000;
  //max_tokens = 0;
  //max_coefficient = 0;
  double timeout = 120.0;

  auto S = [&]() -> td::Status {
    auto b = nlohmann::json::parse(payload.as_slice().begin(), payload.as_slice().end(), nullptr, false, false);

    if (b.is_discarded()) {
      return td::Status::Error(ton::ErrorCode::protoviolation, "expected json object");
    }
    if (!b.is_object()) {
      return td::Status::Error(ton::ErrorCode::protoviolation, "expected json object");
    }

    if (!b.contains("model") || !b["model"].is_string()) {
      return td::Status::Error(ton::ErrorCode::protoviolation, "missing field 'model'");
    }
    model = b["model"].get<std::string>();
    if (b.contains("max_tokens")) {
      if (!b["max_tokens"].is_number_unsigned()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "field 'max_tokens' must be positive integer");
      }
      max_tokens = b["max_tokens"].get<td::int32>();
    }
    if (b.contains("max_completion_tokens")) {
      if (!b["max_completion_tokens"].is_number_unsigned()) {
        return td::Status::Error(ton::ErrorCode::protoviolation,
                                 "field 'max_completion_tokens' must be positive integer");
      }
      max_tokens = b["max_completion_tokens"].get<td::int32>();
      b.erase("max_tokens");
    }
    if (b.contains("max_coefficient")) {
      if (!b["max_coefficient"].is_number_unsigned()) {
        return td::Status::Error(ton::ErrorCode::protoviolation,
                                 "field 'max_coefficient' must be non-negative integer");
      }
      max_coefficient = b["max_coefficient"].get<td::int32>();
      b.erase("max_coefficient");
    }
    if (b.contains("timeout")) {
      if (!b["timeout"].is_number()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "field 'timeout' must be a number");
      }
      timeout = b["timeout"].get<double>();
      b.erase("timeout");
    }
    if (b.contains("enable_debug")) {
      if (!b["enable_debug"].is_boolean()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "field 'enable_debug' must be a boolean");
      }
      enable_debug_ = b["enable_debug"].get<bool>();
      b.erase("enable_debug");
    }
    if (b.contains("request_guid")) {
      if (!b["request_guid"].is_string()) {
        return td::Status::Error(ton::ErrorCode::protoviolation, "field 'request_guid' must be a string");
      }
      ext_request_id_ = td::sha256_bits256(b["request_guid"].get<std::string>());
      b.erase("request_guid");
    }
    payload = td::BufferSlice(b.dump());
    return td::Status::OK();
  }();

  if (S.is_error()) {
    return return_error(S.move_as_error_prefix("failed to parse request: "), nullptr);
  }

  auto r = in_request_->store_tl(td::Bits256::zero());
  auto req = cocoon::cocoon_api::make_object<cocoon_api::http_request>(
      r->method_, r->url_, r->http_version_, std::vector<ton::tl_object_ptr<cocoon_api::http_header>>(),
      std::move(payload));
  for (auto &h : r->headers_) {
    if (h->name_ == "Content-Length" || h->name_ == "Transfer-Encoding" || h->name_ == "Connection") {
      continue;
    }
    req->headers_.push_back(cocoon::cocoon_api::make_object<cocoon_api::http_header>(h->name_, h->value_));
  }
  if (req->payload_.size()) {
    req->headers_.push_back(
        cocoon::cocoon_api::make_object<cocoon_api::http_header>("Content-Length", PSTRING() << req->payload_.size()));
  }

  auto request_data = cocoon::serialize_tl_object(req, true);

  td::BufferSlice request_data_wrapped;
  if (proto_version_ == 0) {
    request_data_wrapped = cocoon::create_serialize_tl_object<cocoon_api::client_runQuery>(
        model, std::move(request_data), max_coefficient, (int)max_tokens, timeout * 0.95, request_id_,
        min_config_version_);
  } else {
    request_data_wrapped = cocoon::create_serialize_tl_object<cocoon_api::client_runQueryEx>(
        model, std::move(request_data), max_coefficient, (int)max_tokens, timeout * 0.95, request_id_,
        min_config_version_, 1, enable_debug_);
  }

  td::actor::send_closure(client_runner_, &ClientRunner::send_message_to_connection, proxy_connection_id_,
                          std::move(request_data_wrapped));

  in_request_ = nullptr;
  in_payload_ = nullptr;
}

void ClientRunningRequest::process_answer_ex_impl(cocoon_api::client_queryAnswerEx &ans) {
  if (!promise_) {
    LOG(ERROR) << "client request " << request_id_.to_hex() << ": received duplicate answer";
    return;
  }

  LOG(DEBUG) << "client request " << request_id_.to_hex() << ": received answer";

  received_answer_at_unix_ = td::Clocks::system();

  auto response = fetch_tl_object<cocoon_api::http_response>(std::move(ans.answer_), true).move_as_ok();

  stats()->answer_bytes_sent += (double)response->payload_.size();

  auto res =
      ton::http::HttpResponse::create(response->http_version_, response->status_code_, response->reason_, false, false)
          .move_as_ok();
  for (auto &x : response->headers_) {
    ton::http::HttpHeader h{x->name_, x->value_};
    res->add_header(std::move(h));
  }

  // Add client timing headers using Unix timestamps
  res->add_header(
      ton::http::HttpHeader{"X-Cocoon-Client-Start", PSTRING() << td::StringBuilder::FixedDouble(started_at_unix_, 6)});
  res->add_header(ton::http::HttpHeader{"X-Cocoon-Client-End",
                                        PSTRING() << td::StringBuilder::FixedDouble(td::Clocks::system(), 6)});

  res->set_keep_alive(keep_alive_);
  res->complete_parse_header().ensure();

  bool is_completed = ans.flags_ & 1;

  out_payload_ = res->create_empty_payload().move_as_ok();
  if (response->payload_.size() > 0) {
    payload_parts_++;
    payload_bytes_ += response->payload_.size();

    add_payload_part(std::move(response->payload_), is_completed, ans.final_info_);
  }
  out_payload_->flush();

  promise_.set_value(std::make_pair(std::move(res), out_payload_));

  if (is_completed) {
    finish_request(true, std::move(ans.final_info_));
  }
}

void ClientRunningRequest::process_answer(ton::tl_object_ptr<cocoon_api::client_queryAnswer> ans) {
  ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info;
  if (ans->is_completed_) {
    final_info =
        ton::create_tl_object<cocoon_api::client_queryFinalInfo>(0, std::move(ans->request_tokens_used_), "", "");
  }
  bool has_final_info = final_info != nullptr;
  process_answer_ex_impl(*ton::create_tl_object<cocoon_api::client_queryAnswerEx>(
      ans->request_id_, std::move(ans->answer_), has_final_info ? 1 : 0, std::move(final_info)));
}

void ClientRunningRequest::process_answer_ex_impl(cocoon_api::client_queryAnswerErrorEx &ans) {
  LOG(DEBUG) << "client request " << request_id_.to_hex() << ": received error";

  if (promise_) {
    return_error(td::Status::Error(ans.error_code_, ans.error_), std::move(ans.final_info_));
  } else {
    finish_request(false, std::move(ans.final_info_));
  }
}

void ClientRunningRequest::process_answer_error(ton::tl_object_ptr<cocoon_api::client_queryAnswerError> ans) {
  ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info =
      ton::create_tl_object<cocoon_api::client_queryFinalInfo>(0, std::move(ans->request_tokens_used_), "", "");
  process_answer_ex_impl(*ton::create_tl_object<cocoon_api::client_queryAnswerErrorEx>(
      ans->request_id_, ans->error_code_, ans->error_, 1, std::move(final_info)));
}

void ClientRunningRequest::process_answer_ex_impl(cocoon_api::client_queryAnswerPartEx &ans) {
  if (promise_) {
    LOG(ERROR) << "client request " << request_id_.to_hex() << ": received payload part before answer";
    return;
  }

  LOG(DEBUG) << "client request " << request_id_.to_hex() << ": received payload part";
  payload_parts_++;
  payload_bytes_ += ans.answer_.size();
  stats()->answer_bytes_sent += (double)ans.answer_.size();

  bool is_completed = ans.flags_ & 1;
  add_payload_part(std::move(ans.answer_), is_completed, ans.final_info_);
  out_payload_->flush();

  if (is_completed) {
    finish_request(true, std::move(ans.final_info_));
  }
}

void ClientRunningRequest::process_answer_part(ton::tl_object_ptr<cocoon_api::client_queryAnswerPart> ans) {
  ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info;
  if (ans->is_completed_) {
    final_info =
        ton::create_tl_object<cocoon_api::client_queryFinalInfo>(0, std::move(ans->request_tokens_used_), "", "");
  }
  bool has_final_info = final_info != nullptr;
  process_answer_ex_impl(*ton::create_tl_object<cocoon_api::client_queryAnswerPartEx>(
      ans->request_id_, std::move(ans->answer_), has_final_info ? 1 : 0, std::move(final_info)));
}

void ClientRunningRequest::process_answer_part_error(ton::tl_object_ptr<cocoon_api::client_queryAnswerPartError> ans) {
  ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info =
      ton::create_tl_object<cocoon_api::client_queryFinalInfo>(0, std::move(ans->request_tokens_used_), "", "");
  process_answer_ex_impl(*ton::create_tl_object<cocoon_api::client_queryAnswerErrorEx>(
      ans->request_id_, ans->error_code_, ans->error_, 1, std::move(final_info)));
}

void ClientRunningRequest::return_error_str(td::int32 ton_error_code, std::string error,
                                            ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info) {
  LOG(WARNING) << "client request " << request_id_.to_hex() << ": sending error: " << error;
  CHECK(promise_);

  td::int32 error_code;
  std::string error_string;
  switch (ton_error_code) {
    case ton::ErrorCode::timeout:
      error_code = ton::http::HttpStatusCode::status_gateway_timeout;
      error_string = "Gateway Timeout";
      break;
    case ton::ErrorCode::notready:
      error_code = ton::http::HttpStatusCode::status_bad_gateway;
      error_string = "Bad Gateway";
      break;
    default:
      error_code = ton::http::HttpStatusCode::status_internal_server_error;
      error_string = "Internal Server Error";
      break;
  }

  td::StringBuilder sb;
  nlohmann::json v = nlohmann::json::object();
  v["error_code"] = ton_error_code;
  v["error"] = error;
  if (final_info && final_info->proxy_debug_.size() > 0) {
    v["proxy"] = nlohmann::json::parse(final_info->proxy_debug_);
  }
  if (final_info && final_info->worker_debug_.size() > 0) {
    v["worker"] = nlohmann::json::parse(final_info->worker_debug_);
  }
  if (enable_debug_) {
    v["client"] = generate_client_debug_inner();
  }
  sb << v.dump();

  auto data = sb.as_cslice();
  auto response = ton::http::HttpResponse::create("HTTP/1.0", error_code, error_string, false, false).move_as_ok();
  response->add_header(ton::http::HttpHeader{"Content-Length", PSTRING() << data.size()});
  response->complete_parse_header();
  out_payload_ = response->create_empty_payload().move_as_ok();
  out_payload_->add_chunk(td::BufferSlice(td::Slice(data)));
  promise_.set_value(std::make_pair(std::move(response), out_payload_));

  finish_request(false, nullptr);
}

void ClientRunningRequest::finish_request(bool is_success,
                                          ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> final_info) {
  LOG(INFO) << "client request " << request_id_.to_hex() << ": completed: success=" << (is_success ? "YES" : "NO")
            << " time=" << run_time() << " payload_parts=" << payload_parts_ << " payload_bytes=" << payload_bytes_;
  if (is_success) {
    stats()->requests_success++;
  } else {
    stats()->requests_failed++;
  }
  stats()->total_requests_time += run_time();

  CHECK(!promise_);
  out_payload_->complete_parse();

  td::actor::send_closure(client_runner_, &ClientRunner::finish_request, request_id_, proxy_);

  stop();
}

const std::shared_ptr<ClientStats> ClientRunningRequest::stats() const {
  return client_runner_.get_actor_unsafe().stats();
}

nlohmann::json ClientRunningRequest::generate_client_debug_inner() {
  nlohmann::json v;
  v["type"] = "client_stats";
  v["start_time"] = started_at_unix_;
  v["answer_receive_start_at"] = received_answer_at_unix_;
  v["answer_receive_end_at"] = td::Clocks::system();
  return v;
}

void ClientRunningRequest::add_last_payload_part_with_debug(
    td::BufferSlice part, const ton::tl_object_ptr<cocoon_api::client_queryFinalInfo> &info) {
  nlohmann::json v = nlohmann::json::object();
  if (info && info->proxy_debug_.size() > 0) {
    v["proxy"] = nlohmann::json::parse(info->proxy_debug_);
  }
  if (info && info->worker_debug_.size() > 0) {
    v["worker"] = nlohmann::json::parse(info->worker_debug_);
  }
  v["client"] = generate_client_debug_inner();

  if (enable_debug_ && !ext_request_id_.is_zero()) {
    td::actor::send_closure(client_runner_, &ClientRunner::add_request_debug_info, ext_request_id_, v.dump());
  }

  if (part.size() == 0) {
    nlohmann::json w;
    w["debug"] = v;
    out_payload_->add_chunk(td::BufferSlice(w.dump() + "\n"));
    return;
  }

  auto S = part.as_slice();

  static const auto is_ws = [&](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };

  auto SS = S.copy();
  while (SS.size() > 0 && is_ws(SS.back())) {
    SS.remove_suffix(1);
  }

  td::StringBuilder sb;
  std::stringstream ss(S.str());
  bool written = false;
  while (true) {
    try {
      size_t old_pos = ss.tellg();
      nlohmann::json w;
      ss >> w;
      size_t pos = ss.tellg();
      if (pos == old_pos) {
        break;
      }
      if (pos != SS.size()) {
        continue;
      }

      w["debug"] = v;
      out_payload_->add_chunk(td::BufferSlice(S.truncate(old_pos)));
      out_payload_->add_chunk(td::BufferSlice("\n" + w.dump() + "\n"));
      written = true;
      break;
    } catch (...) {
      break;
    }
  }
  if (!written) {
    out_payload_->add_chunk(td::BufferSlice(S));
    nlohmann::json w;
    w["debug"] = v;
    out_payload_->add_chunk(td::BufferSlice(w.dump() + "\n"));
    return;
  }
}

}  // namespace cocoon
