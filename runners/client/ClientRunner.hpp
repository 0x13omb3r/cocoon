#pragma once

#include "ClientProxyConnection.h"
#include "ClientProxyInfo.h"
#include "ClientRunningRequest.h"
#include "ClientStats.h"
#include "auto/tl/cocoon_api.h"
#include "checksum.h"
#include "common/bitstring.h"
#include "runners/BaseRunner.hpp"
#include "td/actor/ActorId.h"
#include "td/actor/PromiseFuture.h"
#include "td/actor/actor.h"
#include "td/utils/Time.h"
#include "td/utils/common.h"
#include "tl/TlObject.h"
#include <memory>

namespace cocoon {

class ClientRunner : public BaseRunner {
 public:
  ClientRunner(std::string engine_config_filename) : BaseRunner(std::move(engine_config_filename)) {
  }

  /* CONST PARAMS */
  static constexpr td::int64 min_tokens_on_contract() {
    return 100000000;
  }
  static constexpr td::int32 min_proto_version() {
    return 1;
  }
  static constexpr td::int32 max_proto_version() {
    return 1;
  }
  static constexpr size_t request_log_size() {
    return 1;
  }

  /* SIMPLE GETTERS */
  const auto &owner_address() const {
    return owner_address_;
  }
  td::Slice secret_string() const {
    return secret_string_.as_slice();
  }
  const auto &secret_hash() const {
    return secret_hash_;
  }
  bool check_proxy_hash() const {
    return check_proxy_hash_;
  }

  /* SIMPLE SETTERS */
  void set_owner_address(block::StdAddress owner_address) {
    owner_address_ = std::move(owner_address);
  }
  void set_secret_string(td::SecureString secret_string) {
    secret_string_ = std::move(secret_string);
    secret_hash_ = td::sha256_bits256(secret_string_.as_slice());
  }
  void enable_check_proxy_hash() {
    check_proxy_hash_ = true;
  }

  /* CHARGE AND TOP UP */
  td::Promise<td::Unit> create_proxy_sc_request_promise(std::shared_ptr<ClientProxyInfo> proxy) {
    return [self_id = actor_id(this), proxy](td::Result<td::Unit> R) {
      R.ensure();
      td::actor::send_closure(self_id, &ClientRunner::proxy_sc_request_completed, proxy);
    };
  }
  void proxy_sc_request_completed(std::shared_ptr<ClientProxyInfo> proxy) {
    //proxy->sc_request_completed();
  }

  /* REQUEST */
  void run_http_request(
      std::unique_ptr<ton::http::HttpRequest> request, std::shared_ptr<ton::http::HttpPayload> payload,
      td::Promise<std::pair<std::unique_ptr<ton::http::HttpResponse>, std::shared_ptr<ton::http::HttpPayload>>>
          promise);
  void run_get_models_request(
      td::Promise<std::pair<std::unique_ptr<ton::http::HttpResponse>, std::shared_ptr<ton::http::HttpPayload>>>
          promise);
  void finish_request(td::Bits256 request_id, std::shared_ptr<ClientProxyInfo> proxy);

  /* ALLOCATORS */
  std::unique_ptr<ProxyOutboundConnection> allocate_proxy_outbound_connection(
      TcpClient::ConnectionId connection_id, TcpClient::TargetId target_id, const RemoteAppType &remote_app_type,
      const td::Bits256 &remote_app_hash) override {
    return std::make_unique<ClientProxyConnection>(this, remote_app_type, remote_app_hash, connection_id, target_id);
  }
  std::unique_ptr<ProxyTarget> allocate_proxy_target(TcpClient::TargetId target_id,
                                                     const td::IPAddress &addr) override {
    return std::make_unique<ProxyTarget>(this, addr, target_id);
  }

  /* INITIALIZATION */
  void load_config(td::Promise<td::Unit> promise) override;
  void custom_initialize(td::Promise<td::Unit> promise) override;

  /* CRON */
  void alarm() override;

  /* INBOUND MESSAGE HANDLERS */
  void receive_message(TcpClient::ConnectionId connection_id, td::BufferSlice query) override;
  void receive_query(TcpClient::ConnectionId connection_id, td::BufferSlice query,
                     td::Promise<td::BufferSlice> promise) override {
  }
  void receive_http_request(
      std::unique_ptr<ton::http::HttpRequest> request, std::shared_ptr<ton::http::HttpPayload> payload,
      td::Promise<std::pair<std::unique_ptr<ton::http::HttpResponse>, std::shared_ptr<ton::http::HttpPayload>>> promise)
      override;

  /* PROXY DB*/
  td::Result<std::shared_ptr<ClientProxyInfo>> register_proxy(
      TcpClient::ConnectionId connection_id, const td::Bits256 &proxy_public_key,
      const block::StdAddress &proxy_owner_address, const block::StdAddress &proxy_sc_address,
      const block::StdAddress &client_sc_address, ton::tl_object_ptr<cocoon_api::proxy_SignedPayment> payment);
  void update_proxy_payment_status(const std::string &proxy_sc_address_str, td::BufferSlice info);

  /* CONTROL */
  td::Status cmd_close(const std::string &proxy_sc_address_str);
  td::Status cmd_top_up(const std::string &proxy_sc_address_str);
  td::Status cmd_withdraw(const std::string &proxy_sc_address_str);

  /* HTTP */
  std::string wrap_short_answer_to_http(const std::string &text) {
    td::StringBuilder sb;
    sb << "<!DOCTYPE html>\n";
    sb << "<html><body>\n";
    sb << text << "<br/>\n";
    sb << "<a href=\"/stats\">return to stats</a>\n";
    sb << "</html></body>\n";
    return sb.as_cslice().str();
  };
  std::string wrap_short_answer_to_http(td::Status error) {
    if (error.is_ok()) {
      return wrap_short_answer_to_http("Request sent");
    } else {
      return wrap_short_answer_to_http(PSTRING() << "failed: " << error);
    }
  }
  std::string http_generate_main();
  std::string http_generate_json_stats();
  std::string http_charge(const std::string &proxy_sc_address);
  std::string http_close(const std::string &proxy_sc_address) {
    return wrap_short_answer_to_http(cmd_close(proxy_sc_address));
  }
  std::string http_top_up(const std::string &proxy_sc_address) {
    return wrap_short_answer_to_http(cmd_top_up(proxy_sc_address));
  }
  std::string http_withdraw(const std::string &proxy_sc_address) {
    return wrap_short_answer_to_http(cmd_withdraw(proxy_sc_address));
  }
  std::string http_get_request_debug_info(const std::string &request_guid) {
    auto id = td::sha256_bits256(request_guid);
    auto it = request_debug_info_.find(id);
    if (it != request_debug_info_.end()) {
      return it->second;
    } else {
      return "{}";
    }
  }

  const auto &stats() const {
    return stats_;
  }

  void add_request_debug_info(td::Bits256 request_id, std::string value) {
    if (request_debug_info_.count(request_id)) {
      return;
    }
    request_debug_info_.emplace(request_id, std::move(value));
    request_debug_info_lru_.push_back(request_id);

    if (request_debug_info_lru_.size() > request_log_size()) {
      CHECK(request_debug_info_.erase(request_debug_info_lru_.front()));
      request_debug_info_lru_.pop_front();
    }
  }

 private:
  block::StdAddress owner_address_;
  td::SecureString secret_string_;
  td::Bits256 secret_hash_ = td::Bits256::zero();
  std::map<std::string, std::shared_ptr<ClientProxyInfo>> proxies_;
  std::map<td::Bits256, td::actor::ActorId<ClientRunningRequest>> running_queries_;
  std::map<td::Bits256, std::string> request_debug_info_;
  std::list<td::Bits256> request_debug_info_lru_;

  std::unique_ptr<td::Ed25519::PrivateKey> wallet_private_key_;
  td::Bits256 wallet_public_key_;

  td::Timestamp next_payment_compare_at_;
  td::Timestamp next_update_balances_at_;
  td::uint32 params_version_{0};

  bool check_proxy_hash_{false};

  std::shared_ptr<ClientStats> stats_ = std::make_shared<ClientStats>();
};

}  // namespace cocoon
