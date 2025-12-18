/*
 * health-client.cpp
 * 
 * Client tool to query health-monitor running in TDX guest via vsock.
 */

#include "td/utils/logging.h"
#include "td/utils/OptionParser.h"
#include "td/utils/Status.h"
#include "td/utils/buffer.h"
#include "td/utils/port/SocketFd.h"
#include "td/actor/actor.h"
#include "td/net/Pipe.h"
#include "td/net/utils.h"
#include "utils.h"

#include <string>
#include <sstream>

namespace cocoon {

class HealthClient final : public td::TaskActor<td::Unit> {
 public:
  struct Config {
    td::uint32 cid_;
    td::int32 port_;
    std::string request_;
  };

  explicit HealthClient(Config config) : config_(std::move(config)) {
  }

 private:
  Config config_;
  td::SocketPipe pipe_;
  bool request_sent_ = false;

  td::Task<Action> task_loop_once() override {
    if (!pipe_) {
      LOG(INFO) << "Connecting to guest CID " << config_.cid_ << " port " << config_.port_;
      auto socket = co_await td::SocketFd::open_vsock(config_.cid_, config_.port_);
      pipe_ = td::make_socket_pipe(std::move(socket));
      pipe_.subscribe();
      LOG(INFO) << "Connected to health-monitor";
    }

    co_await pipe_.flush_read();

    // Send request once
    if (!request_sent_) {
      co_await framed_write(pipe_.output_buffer(), config_.request_);
      request_sent_ = true;
      LOG(INFO) << "Request sent: " << config_.request_;
    }

    co_await pipe_.flush_write();

    // Try to read response
    td::BufferSlice response;
    auto needed = co_await framed_read(pipe_.input_buffer(), response, 100 * (1 << 20));
    if (needed > 0) {
      // Need more data
      co_return Action::KeepRunning;
    }

    // Got complete response - just print it
    LOG(PLAIN) << response.as_slice().str();
    co_return Action::Finish;
  }

  td::Task<td::Unit> finish(td::Status status) override {
    if (status.is_error()) {
      LOG(ERROR) << "Error: " << status.error();
    }
    td::actor::SchedulerContext::get().stop();
    co_return td::Unit{};
  }
};

}  // namespace cocoon

int main(int argc, char** argv) {
  SET_VERBOSITY_LEVEL(VERBOSITY_NAME(WARNING));

  constexpr int SCHEDULER_THREADS = 1;
  constexpr int SCHEDULER_TIMEOUT_MS = 10;
  constexpr td::int32 DEFAULT_VSOCK_PORT = 9999;
  constexpr td::uint32 DEFAULT_CID = 4;

  td::uint32 cid = DEFAULT_CID;
  td::int32 vsock_port = DEFAULT_VSOCK_PORT;

  // Helper to parse instance specification (e.g., "worker", "worker:3", "proxy:2")
  auto parse_instance = [&](td::Slice instance_str) -> td::Status {
    std::string instance = instance_str.str();

    // Find the base type and instance number
    std::string type;
    int instance_num = 0;  // default to first instance

    size_t colon_pos = instance.find(':');
    if (colon_pos != std::string::npos) {
      type = instance.substr(0, colon_pos);
      std::string num_str = instance.substr(colon_pos + 1);
      TRY_RESULT(num, td::to_integer_safe<int>(num_str));
      if (num < 0) {
        return td::Status::Error("Instance number must be positive");
      }
      instance_num = num;
    } else {
      type = instance;
    }

    // Map type to base CID
    td::uint32 base_cid;
    if (type == "worker" || type == "w") {
      base_cid = 6;
    } else if (type == "proxy" || type == "p") {
      base_cid = 7;
    } else if (type == "client" || type == "c") {
      base_cid = 4;
    } else {
      return td::Status::Error(PSLICE() << "Unknown instance type: " << type << ". Use 'worker', 'proxy', or 'client'");
    }

    cid = base_cid + instance_num * 10;
    return td::Status::OK();
  };

  td::OptionParser option_parser;
  option_parser.add_checked_option('c', "cid", "Guest CID to connect to", [&](td::Slice cid_str) {
    TRY_RESULT(parsed_cid, td::to_integer_safe<td::uint32>(cid_str));
    if (parsed_cid == 0) {
      return td::Status::Error("Invalid CID");
    }
    cid = parsed_cid;
    return td::Status::OK();
  });

  option_parser.add_checked_option('i', "instance", "Instance name (e.g., worker, worker:3, proxy:2, client)",
                                   parse_instance);

  option_parser.add_checked_option('p', "port", "VSOCK port to connect to", [&](td::Slice port_str) {
    TRY_RESULT(port, td::to_integer_safe<td::int32>(port_str));
    if (port <= 0 || port > 65535) {
      return td::Status::Error("Invalid port number");
    }
    vsock_port = port;
    return td::Status::OK();
  });

  option_parser.add_option('h', "help", "Show this help message", [&]() {
    LOG(PLAIN) << "Usage: " << argv[0] << " [options] <command> [args...]";
    LOG(PLAIN) << "";
    LOG(PLAIN) << "Commands:";
    LOG(PLAIN) << "  status [service]            - Get overall health and service status";
    LOG(PLAIN) << "  sys                         - Get system metrics (CPU, memory, disk, network)";
    LOG(PLAIN) << "  svc <service>               - Get detailed service info with recent logs";
    LOG(PLAIN) << "  logs <service> [lines]      - Get service logs (default 100 lines)";
    LOG(PLAIN) << "  tdx                         - Get TDX attestation status (image hash + RTMRs)";
    LOG(PLAIN) << "  eventlog                    - Get TDX event log (CCEL ACPI table) with RTMR replay";
    LOG(PLAIN) << "  gpu                         - Get GPU metrics (utilization, memory, temp, power)";
    LOG(PLAIN) << "  all                         - Get all metrics in one view";
    LOG(PLAIN) << "";
    LOG(PLAIN) << "Options:";
    LOG(PLAIN) << option_parser;
    LOG(PLAIN) << "";
    LOG(PLAIN) << "Instance Types:";
    LOG(PLAIN) << "  worker, w      - Worker instances (CID 6, 16, 26, ...)";
    LOG(PLAIN) << "  proxy, p       - Proxy instances (CID 7, 17, 27, ...)";
    LOG(PLAIN) << "  client, c      - Client instances (CID 4, 14, 24, ...)";
    LOG(PLAIN) << "";
    LOG(PLAIN) << "Examples:";
    LOG(PLAIN) << "  " << argv[0] << " --instance worker status";
    LOG(PLAIN) << "  " << argv[0] << " -i worker:3 sys";
    LOG(PLAIN) << "  " << argv[0] << " -i worker gpu";
    LOG(PLAIN) << "  " << argv[0] << " -i worker tdx";
    LOG(PLAIN) << "  " << argv[0] << " -i worker eventlog";
    LOG(PLAIN) << "  " << argv[0] << " -i worker svc cocoon-vllm";
    LOG(PLAIN) << "  " << argv[0] << " -i proxy:2 logs cocoon-vllm 200";
    LOG(PLAIN) << "  " << argv[0] << " -i worker all";
    LOG(PLAIN) << "  " << argv[0] << " --cid 6 status cocoon-worker-runner";
    std::_Exit(0);
  });

  option_parser.set_description("health-client: query health-monitor in guest via vsock");

  auto r_args = option_parser.run(argc, argv, -1);
  if (r_args.is_error()) {
    LOG(ERROR) << r_args.error();
    LOG(ERROR) << option_parser;
    return 1;
  }

  // Collect remaining arguments as the command
  auto remaining = r_args.move_as_ok();
  if (remaining.empty()) {
    LOG(ERROR) << "No command specified. Use --help for usage information.";
    return 1;
  }

  // Build request string from remaining arguments
  std::ostringstream request_stream;
  for (size_t i = 0; i < remaining.size(); ++i) {
    if (i > 0)
      request_stream << " ";
    request_stream << remaining[i];
  }
  std::string request = request_stream.str();

  LOG(INFO) << "Connecting to CID " << cid << " port " << vsock_port;
  LOG(INFO) << "Request: " << request;

  td::actor::Scheduler sched({SCHEDULER_THREADS});

  sched.run_in_context([&] {
    cocoon::HealthClient::Config config{.cid_ = cid, .port_ = vsock_port, .request_ = request};
    td::spawn_task_actor<cocoon::HealthClient>("HealthClient", std::move(config)).detach("HealthClient");
  });

  sched.start();

  // Run until completion
  while (sched.run(SCHEDULER_TIMEOUT_MS)) {
    // Continue processing
  }

  return 0;
}
